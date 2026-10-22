#!/usr/bin/env python3
"""End-to-end MPCache controller on CogVideoX (V100): online J-hat + budgeted receding-horizon planning.

Compared at the SAME number N of exact denoiser evaluations (matched NFE):
  reuse   last exact output, evaluated on the static TeaCache schedule (TeaCache-like reuse)
  ex1     first-order temporal extrapolation on the same schedule
  tea_mp  TeaCache schedule + our state-model correction (the G0b forecaster)
  mpc     our controller: it chooses WHEN to evaluate (budget) as well as WHAT to forecast,
          using the same online state model.

Planner (dual-ascent quota).  At step i with budget b = N - spent left and r = T - i steps left:
  b >= r          -> must evaluate (no room left to skip)
  b == 0          -> must skip
  otherwise       score_i = || beta * TR(A d_i) || * g_i,   g_i = (T - i) / T
                  evaluate iff score_i > lam, with lam adapted toward the linear quota N(i+1)/T.
The score is the model's predicted deviation from ex1, weighted by the remaining propagation window:
early skips have more steps left to amplify, so they are charged more.

Metric: terminal-latent relative L2 against the all-exact run of the same prompt/seed.
Development protocol: CogVideoX-2B, 17 frames (5 latent), 480x720, DDIM-30, CFG 6.
Production protocol: 49 frames (13 latent), DDIM-50, set with --frames 49.

usage:
  python mpcache_controller.py --frames 17 --Ns 10,13,17 --nprompt 8 --seeds 3000,3001 \
      --out /root/viewdit/results/video/mpc_20261020
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/viewdit')
MODEL = ROOT / 'weights/Wan2.1-T2V-1.3B'
CFG = 5.0
C = 16
SHIFT = 5.0
TEA_COEF = [-3.10658903e+01, 2.54732368e+01, -5.92380459e+00, 1.75769064e+00, -3.61568434e-03]
GAMMA, LAM_D, LAM_F, ACLIP, KAPPA, BETA0 = 0.5, 0.1, 0.1, 3.0, 1.0, 0.5
W1 = 0.3
PGATE = 0.02
BIAS = False
PROMPTS = [
    'A white swan swims slowly across a calm blue pond, gentle ripples follow the swan, steady camera, realistic wildlife footage.',
    'A red bus drives slowly along a city street, buildings in the background, steady camera, realistic footage.',
    'A brown bear walks along a river bank in a dense forest, realistic wildlife documentary.',
    'A panda eats bamboo while sitting in a lush green bamboo forest, close-up.',
    'Fireworks explode over a city skyline at night, reflections in the river.',
    'A cat stretches and yawns on a sunny windowsill, soft natural light.',
    'A surfer rides a large ocean wave, spray in the air, aerial drone shot.',
    'A horse gallops across an open meadow at golden hour, dramatic lighting.',
]

# Prompt list resolution: $MPCACHE_PROMPTS, then the RA-CFGCache list at
# /root/viewdit/resources/prompts/prompt_video.txt (the paper protocol), then the
# copy shipped next to this script, then the builtin 8 kept for old runs.
_PROMPT_CANDS = []
_env_p = os.environ.get('MPCACHE_PROMPTS', '')
if _env_p:
    _PROMPT_CANDS.append(Path(_env_p))
_PROMPT_CANDS += [Path('/root/viewdit/resources/prompts/prompt_video.txt'),
                  Path(__file__).resolve().parent / 'prompts_100.txt']
_PROMPT_FILE = next((p for p in _PROMPT_CANDS if p.exists()), None)
if _PROMPT_FILE is not None:
    PROMPTS = [ln.strip() for ln in _PROMPT_FILE.read_text().splitlines() if ln.strip()]
    print(f'[prompts] {len(PROMPTS)} from {_PROMPT_FILE}', flush=True)

FRAMES = {33: dict(lt=9, ph=60, pw=104, steps=50), 81: dict(lt=21, ph=60, pw=104, steps=50)}
MAX_GAP = 4
GAMMA_PACE = 1.0
RESERVE = 1
TOK_Q = -1.0
MODEL_KIND = 'diag'
GATE_AGE = 2
TRUST = -1.0
PEAK_LOOK = 6
PEAK_FRAC = 0.9
LT, PH, PW, NSTEP = 9, 60, 104, 50


def tok2lat(vt):
    o = vt.view(1, LT, PH // 2, PW // 2, C, 1, 2, 2)
    return o.permute(0, 4, 1, 2, 5, 6, 3, 7).reshape(1, C, LT, PH, PW)


def lat2tok(lat):
    return lat.float().view(1, C, LT, PH // 2, 2, PW // 2, 2).permute(0, 2, 3, 5, 1, 4, 6).reshape(
        LT * (PH // 2) * (PW // 2), 64)


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


class StateModel:
    """Causal structured estimate of J on the token-channel space + self-check gate (G0b)."""

    def __init__(self, kind='diag', beta_mode='gate', innov_norm='none'):
        self.kind = kind
        self.beta_mode = beta_mode
        self.innov_norm = innov_norm
        self.ratio_ema = 1.0
        self.Sdd = torch.zeros(64, device='cuda', dtype=torch.float64)
        self.Sde = torch.zeros(64, device='cuda', dtype=torch.float64)
        self.G = torch.zeros(64, 64, device='cuda', dtype=torch.float64)
        self.Se = torch.zeros(64, device='cuda', dtype=torch.float64)
        self.Sw = 0.0
        self.R = torch.zeros(64, 64, device='cuda', dtype=torch.float64)
        self.n = 0
        self.beta = BETA0
        self.log = []

    def correction(self, d):
        if self.n == 0:
            return torch.zeros_like(d)
        if self.kind == 'full':
            lam = LAM_F * self.G.diagonal().mean()
            A = torch.linalg.solve(self.G + lam * torch.eye(64, device='cuda', dtype=torch.float64), self.R)
            return (d.double() @ A).float()
        a = (self.Sde / ((1 + LAM_D) * self.Sdd + 1e-12)).clamp(-ACLIP, ACLIP)
        out = d * a.float()
        if BIAS and self.Sw > 1e-9:
            out = out + (self.Se / self.Sw).float().expand_as(out)
        return out

    def forecast(self, v_ex1, v_s, d, cap_base=None):
        corr = self.correction(d)
        cap = KAPPA * (cap_base if cap_base is not None else float((v_ex1 - v_s).norm()))
        cn = float(corr.norm())
        if cn > cap:
            corr = corr * (cap / (cn + 1e-12))
        return v_ex1 + self.beta * corr

    def observe(self, d, e, v_ex1_k, v_prev, cap_base=None):
        if self.n > 0:
            pred = self.forecast(v_ex1_k, v_prev, d, cap_base)
            err_m, err_e = float((pred - (v_ex1_k + e)).norm()), float(e.norm())
            self.beta = min(1.0, self.beta + 0.25) if err_m < err_e else self.beta * 0.5
            self.log.append((err_m / (err_e + 1e-12), self.beta))
        dd, ee = d.double(), e.double()
        self.Sdd = GAMMA * self.Sdd + (dd * dd).sum(0)
        self.Se = GAMMA * self.Se + ee.mean(0)
        self.Sw = GAMMA * self.Sw + 1.0
        self.Sde = GAMMA * self.Sde + (dd * ee).sum(0)
        self.G = GAMMA * self.G + dd.T @ dd
        self.R = GAMMA * self.R + dd.T @ ee
        self.n += 1


def main():
    from diffusers import WanTransformer3DModel, UniPCMultistepScheduler, FlowMatchEulerDiscreteScheduler
    from transformers import AutoTokenizer, UMT5EncoderModel

    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=33, choices=[33, 81])
    ap.add_argument('--seeds', default='3000,3001')
    ap.add_argument('--nprompt', type=int, default=8)
    ap.add_argument('--prompt-offset', type=int, default=0)
    ap.add_argument('--Ns', default='10,13,17')
    ap.add_argument('--planners', default='reuse,ex1,tea_mp,mpc,mpc_reuse')
    ap.add_argument('--out', default=str(ROOT / 'results/video/mpc_20261020'))
    ap.add_argument('--max-gap', type=int, default=4, help='hard cap on consecutive skipped steps (mpc)')
    ap.add_argument('--reserve-last', type=int, default=1, help='evaluations reserved for the final steps')
    ap.add_argument('--model-kind', default='diag', choices=['diag','full'])
    ap.add_argument('--gate-age', type=int, default=2)
    ap.add_argument('--trust', type=float, default=-1.0)
    ap.add_argument('--kappa', type=float, default=1.0)
    ap.add_argument('--w1', type=float, default=0.3)
    ap.add_argument('--rls-gamma', type=float, default=0.5)
    ap.add_argument('--bias', type=int, default=0)
    ap.add_argument('--pgate', type=float, default=0.02)
    ap.add_argument('--gamma', type=float, default=0.6, help='budget pacing exponent: quota_i = N*((i+1)/T)**gamma')
    ap.add_argument('--solver', default='euler', choices=['euler', 'unipc'])
    ap.add_argument('--sched', default='uniform', choices=['uniform', 'sigma'])
    ap.add_argument('--seed-from-prompt', action='store_true',
                    help='paper protocol: seed == prompt index')
    ap.add_argument('--negative', default='',
                    help='negative prompt (paper protocol: "animation")')
    ap.add_argument('--no-full', action='store_true',
                    help='skip the all-exact reference (use RA Full videos as the baseline)')
    ap.add_argument('--save-video', action='store_true')
    ap.add_argument('--save-op', default='tea_gap', help='comma separated ops to render')
    ap.add_argument('--save-n', type=int, default=13)
    ap.add_argument('--video-dir', default='')
    ap.add_argument('--token-gate', type=float, default=-1.0,
                    help='per-token curvature gate: apply the correction only to tokens with '
                         'kappa=||a||/||v||^2 above this quantile (0..1); <0 disables')
    ap.add_argument('--sanity', action='store_true')
    a = ap.parse_args()

    global LT, PH, PW, NSTEP, MAX_GAP, GAMMA_PACE, RESERVE
    cfgf = FRAMES[a.frames]
    LT, PH, PW, NSTEP = cfgf['lt'], cfgf['ph'], cfgf['pw'], cfgf['steps']
    MAX_GAP = a.max_gap
    GAMMA_PACE = a.gamma
    global MODEL_KIND, GATE_AGE, TRUST, KAPPA, W1, GAMMA, BIAS, PGATE, TOK_Q
    MODEL_KIND = a.model_kind
    GATE_AGE = a.gate_age
    TRUST = a.trust
    KAPPA = a.kappa
    W1 = a.w1
    GAMMA = a.rls_gamma
    BIAS = bool(a.bias)
    PGATE = a.pgate
    TOK_Q = a.token_gate
    RESERVE = a.reserve_last

    seeds = [int(s) for s in a.seeds.split(',')]
    Ns = [int(n) for n in a.Ns.split(',')]
    planners = [p for p in a.planners.split(',') if p]
    OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    LC = OUT / 'closed.jsonl'
    done = set()
    if LC.exists():
        for line in LC.read_text().splitlines():
            r = json.loads(line); done.add((r['prompt'], r['seed'], r['N'], r['op']))

    emb_path = OUT / f'embeddings_fp16_eager_{a.frames}f.pt'
    lo = max(0, a.prompt_offset)
    hi = min(len(PROMPTS), lo + a.nprompt)
    if hi <= lo:
        raise SystemExit(f'empty prompt range: offset={a.prompt_offset} nprompt={a.nprompt} total={len(PROMPTS)}')
    need = list(range(lo, hi))
    if emb_path.exists():
        cache = torch.load(emb_path, map_location='cuda', weights_only=True)
        embs, ne = {int(k): v for k, v in cache['embs'].items()}, cache['ne']
    else:
        embs, ne = {}, None
    missing = [p for p in need if p not in embs]
    if missing or ne is None:
        tok = AutoTokenizer.from_pretrained(MODEL / 'tokenizer')
        enc = UMT5EncoderModel.from_pretrained(MODEL / 'text_encoder', torch_dtype=torch.float16).cuda().eval()
        with torch.no_grad():
            for pi in missing:
                ti = tok(PROMPTS[pi], padding='max_length', max_length=512, truncation=True,
                         return_attention_mask=True, return_tensors='pt')
                embs[pi] = enc(ti.input_ids.cuda(), ti.attention_mask.cuda())[0].to(torch.float16)
            if ne is None:
                ti = tok(a.negative, padding='max_length', max_length=512, truncation=True,
                         return_attention_mask=True, return_tensors='pt')
                ne = enc(ti.input_ids.cuda(), ti.attention_mask.cuda())[0].to(torch.float16)
        del enc; torch.cuda.empty_cache()
        torch.save({'embs': {str(k): v for k, v in embs.items()}, 'ne': ne}, emb_path)
    print(f'prompts total={len(PROMPTS)} using [{lo},{hi}) -> {hi - lo} prompts', flush=True)

    d = WanTransformer3DModel.from_pretrained(MODEL / 'transformer', torch_dtype=torch.float16).cuda().eval().requires_grad_(False)

    vae = None
    if a.save_video:
        from diffusers import AutoencoderKLWan
        import imageio
        vae = AutoencoderKLWan.from_pretrained(MODEL / 'vae', torch_dtype=torch.float16).cuda().eval()
        VDIR = Path(a.video_dir) if a.video_dir else (OUT / 'videos')
        VDIR.mkdir(parents=True, exist_ok=True)

        def save_video(z, idx, op):
            sub = VDIR / op
            sub.mkdir(parents=True, exist_ok=True)
            with torch.no_grad():
                v = vae.decode(z.to(torch.float16)).sample
            v = ((v.clamp(-1, 1) + 1) / 2 * 255).byte().cpu().permute(0, 2, 3, 4, 1)[0].numpy()
            imageio.mimsave(sub / f'video_{idx:06d}.mp4', v, fps=16, codec='libx264',
                            macro_block_size=8)
            print(f'[video] {op} {idx:06d} -> {sub}', flush=True)

    if a.solver == 'unipc':
        sch = UniPCMultistepScheduler.from_pretrained(MODEL / 'scheduler')
    else:
        sch = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=SHIFT,
                                              use_dynamic_shifting=False)
    sch.set_timesteps(NSTEP, device='cuda')
    TS = sch.timesteps

    @torch.no_grad()
    def model(lat, pe, t):
        o = d(hidden_states=torch.cat([lat] * 2), encoder_hidden_states=torch.cat([ne, pe]),
              timestep=t.expand(2), return_dict=False)[0].float()
        u, c = o.chunk(2)
        return lat2tok(u + CFG * (c - u))

    def model_both(lat, pe, t):
        o = d(hidden_states=torch.cat([lat] * 2), encoder_hidden_states=torch.cat([ne, pe]),
              timestep=t.expand(2), return_dict=False)[0].float()
        u, c = o.chunk(2)
        return lat2tok(u), lat2tok(c)

    def step(v, t, lat):
        return sch.step(tok2lat(v).to(torch.float16), t, lat, return_dict=False)[0]

    def init(seed):
        sch.set_timesteps(NSTEP, device='cuda')
        g = torch.Generator(device='cuda').manual_seed(seed)
        sig0 = float(getattr(sch, 'init_noise_sigma', 1.0))
        return torch.randn((1, C, LT, PH, PW), generator=g, device='cuda', dtype=torch.float16) * sig0

    lat = init(0)
    print('sanity roundtrip', rel(tok2lat(lat2tok(lat)), lat.float()), flush=True)
    torch.cuda.synchronize(); t0 = time.time(); model(lat, embs[need[0]], TS[min(5, NSTEP - 1)]); torch.cuda.synchronize()
    print(f'one CFG step {time.time() - t0:.2f}s; peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)
    if a.sanity:
        return

    sig = getattr(sch, 'sigmas', None)
    if sig is None:
        dist = [None] + [1.0] * (NSTEP - 1)
    else:
        sig = sig.detach().float().cpu()
        dist = [None] + [float(abs(sig[i] - sig[i - 1]) / (abs(float(sig[i - 1])) + 1e-6)) for i in range(1, NSTEP)]

    def tea_sched(th):
        S, acc = {0, NSTEP - 1}, 0.
        for i in range(1, NSTEP - 1):
            acc += dist[i]
            if acc >= th:
                S.add(i); acc = 0.
        return S

    def tea_for(N):
        lo, hi, best = 0., 50., None
        for _ in range(80):
            mid = (lo + hi) / 2; S = tea_sched(mid)
            if best is None or abs(len(S) - N) < abs(len(best) - N):
                best = S
            if len(S) > N: lo = mid
            elif len(S) < N: hi = mid
            else: return S
        return best
    if a.sched == 'uniform':
        tea = {N: sorted({round(i * (NSTEP - 1) / (N - 1)) for i in range(N)}) for N in Ns}
    else:
        tea = {N: sorted(tea_for(N)) for N in Ns}
    (OUT / 'schedules.json').write_text(json.dumps({str(k): v for k, v in tea.items()}))
    for N in Ns:
        print(f'tea N={N} (|S|={len(tea[N])}): {tea[N]}', flush=True)

    
    def plan_dp(first, last, e_gap, gw, K):
        n = last - first + 1
        INF = float('inf')
        cost = [[0.0] * n for _ in range(n)]
        for j in range(n):
            for t in range(j + 1, n):
                cost[j][t] = cost[j][t - 1] + e_gap.get(t - j - 1, 0.0) * gw[first + t - 1]
        dp = [[INF] * n for _ in range(K + 1)]
        par = [[-1] * n for _ in range(K + 1)]
        for t in range(n):
            dp[1][t] = cost[0][t]
        for k in range(2, K + 1):
            for t in range(n):
                best, bj = INF, -1
                for j in range(t):
                    v = dp[k - 1][j] + cost[j][t]
                    if v < best:
                        best, bj = v, j
                dp[k][t], par[k][t] = best, bj
        best, bk, bt = INF, 1, n - 1
        for k in range(1, K + 1):
            for t in range(n):
                tail = sum(e_gap.get(u - t, 0.0) * gw[first + u] for u in range(t + 1, n))
                if dp[k][t] + tail < best:
                    best, bk, bt = dp[k][t] + tail, k, t
        chosen = []
        k, t = bk, bt
        while t >= 0 and k >= 1:
            chosen.append(first + t)
            t, k = par[k][t], k - 1
        return set(chosen)

    def run_closed(pe, seed, N, op):
            """One closed-loop run; returns terminal latent and nfe."""
            lat = init(seed)
            _bm = 'smooth' if op in ('mp_smooth', 'mp_r', 'tea_gap1') else 'gate'
            _in = 'r' if op == 'mp_r' else 'none'
            H, sm, nfe, spent = [], StateModel(MODEL_KIND if op in ('mp_reuse', 'tea_gap', 'tea_gap1') else 'diag', _bm, _in), 0, 0
            DEFER = [False]
            sm_u = StateModel('diag')
            sm_c = StateModel('diag')
            lam, seen = None, []
            acc, theta, ema = 0.0, None, None
            PLAN = None
            W = {'reuse': 1.0, 'ex1': 1.0, 'model': 1.0}
            ETA_MIX = 1.0
            Sset = set(tea[N]) if op not in ('mpc', 'mpc_th_in', 'mpc_th_md', 'peak', 'dp') else None
            chosen = []
            def experts(i, x):
                s_, sp_ = H[-1], H[-2]
                rt_ = (i - s_['i']) / (s_['i'] - sp_['i'])
                dv_ = (x - s_['x']) - rt_ * (s_['x'] - sp_['x'])
                v1_ = s_['v'] + rt_ * (s_['v'] - sp_['v'])
                e_ = {'reuse': s_['v'], 'ex1': v1_}
                e_['model'] = sm.forecast(v1_, s_['v'], dv_) if sm.n else v1_
                return e_, dv_, v1_


            def pred_gain(j, xj, s_, sp_):
                """Predicted terminal cost of skipping step j (no denoiser call)."""
                rt_ = (j - s_['i']) / (s_['i'] - sp_['i'])
                dv_ = (xj - s_['x']) - rt_ * (s_['x'] - sp_['x'])
                v1_ = s_['v'] + rt_ * (s_['v'] - sp_['v'])
                corr_ = sm.forecast(v1_, s_['v'], dv_) - v1_
                return ((NSTEP - j) / NSTEP) * float(corr_.norm())

            for i, tt in enumerate(TS):
                x = lat2tok(lat)
                if op == 'dp':
                    r_left, b_left = NSTEP - i, N - spent
                    if b_left <= 0:
                        do = False
                    elif i >= NSTEP - RESERVE:
                        do = True
                    elif b_left >= r_left:
                        do = True
                    elif len(H) < 2 or sm.n < 2:
                        do = True
                    else:
                        if PLAN is None or i >= max(PLAN) or i not in PLAN:
                            s_, sp_ = H[-1], H[-2]
                            e_gap = {0: pred_gain(i, x, s_, sp_)}
                            ls_ = tok2lat(x)
                            for t in range(i + 1, NSTEP - RESERVE):
                                ls_ = step(s_['v'], TS[t], ls_)
                                e_gap[t - i] = pred_gain(t, lat2tok(ls_), s_, sp_)
                            gw = [(NSTEP - t) / NSTEP for t in range(NSTEP)]
                            PLAN = plan_dp(i, NSTEP - RESERVE - 1, e_gap, gw,
                                                    max(1, min(b_left, N - RESERVE - spent)))
                        do = (i in PLAN)
                elif op == 'peak':
                    r_left, b_left = NSTEP - i, N - spent
                    if b_left <= 0:
                        do = False
                    elif i >= NSTEP - RESERVE:
                        do = True
                    elif b_left >= r_left:
                        do = True
                    elif len(H) < 2 or sm.n < 2:
                        do = True
                    elif spent >= max(1, N - RESERVE):
                        do = False
                    else:
                        s_, sp_ = H[-1], H[-2]
                        gain_now = pred_gain(i, x, s_, sp_)
                        xs_, ls_ = x, tok2lat(x)
                        best_ahead = 0.0
                        for j in range(i + 1, min(NSTEP, i + PEAK_LOOK + 1)):
                            ls_ = step(s_['v'], TS[j], ls_)
                            xs_ = lat2tok(ls_)
                            best_ahead = max(best_ahead, pred_gain(j, xs_, s_, sp_))
                        do = gain_now >= PEAK_FRAC * best_ahead
                elif op in ('mpc_th_in', 'mpc_th_md'):
                    # TeaCache-style cumulative threshold, but on our own score and with an
                    # online threshold that is nudged to hit the budget.
                    r_left, b_left = NSTEP - i, N - spent
                    if b_left <= 0:
                        do = False
                    elif i >= NSTEP - RESERVE:
                        do = True
                    elif b_left >= r_left:
                        do = True
                    elif len(H) < 2 or sm.n < 2:
                        do = True
                    elif spent >= max(1, N - RESERVE):
                        do = False                     # keep the reserved evaluations for the tail
                    else:
                        s, sp = H[-1], H[-2]
                        rt = (i - s['i']) / (s['i'] - sp['i'])
                        if op == 'mpc_th_in':
                            sc = float((x - s['x']).norm()) / (float(s['x'].norm()) + 1e-9)
                        else:
                            dvec = (x - s['x']) - rt * (s['x'] - sp['x'])
                            v_ex1 = s['v'] + rt * (s['v'] - sp['v'])
                            corr = sm.forecast(v_ex1, s['v'], dvec) - v_ex1
                            sc = float(corr.norm()) if sm.n else float((v_ex1 - s['v']).norm())
                        # normalise by a running mean so that the accumulated score has unit
                        # mean, then a fixed threshold T/N yields about N evaluations.
                        if ema is None:
                            ema = sc
                        acc += sc / max(ema, 1e-12)
                        ema = 0.9 * ema + 0.1 * sc
                        do = acc >= (NSTEP - RESERVE) / max(1, N - RESERVE)
                    if do:
                        acc = 0.0
                elif op in ('mpc', 'mpc_reuse', 'mpc_in', 'mpc_tm', 'mpc_md'):
                    r_left, b_left = NSTEP - i, N - spent
                    if b_left <= 0:
                        do = False
                    elif i >= NSTEP - RESERVE:
                        do = True                      # always evaluate the final steps
                    elif b_left >= r_left:
                        do = True
                    elif len(H) < 2 or sm.n < 2:
                        do = True                      # warm start: the model cannot score before it is fitted
                    elif i - H[-1]['i'] >= MAX_GAP:
                        do = True                      # never extrapolate farther than MAX_GAP steps
                    elif spent >= math.ceil(max(1, N - RESERVE) * (((i + 1) / max(1, NSTEP - RESERVE)) ** GAMMA_PACE)):
                        do = False                     # hard quota: never run ahead of an even spend
                    else:
                        s, sp = H[-1], H[-2]
                        rt = (i - s['i']) / (s['i'] - sp['i'])
                        dvec = (x - s['x']) - rt * (s['x'] - sp['x'])
                        v_ex1 = s['v'] + rt * (s['v'] - sp['v'])
                        g = (NSTEP - i) / NSTEP
                        base = float((v_ex1 - s['v']).norm())
                        if op == 'mpc_in':
                            rel = float((x - s['x']).norm()) / (float(s['x'].norm()) + 1e-9)
                            score = rel * g
                        elif op == 'mpc_md':
                            corr = sm.forecast(v_ex1, s['v'], dvec) - v_ex1
                            score = (float(corr.norm()) if sm.n else base) * g
                        else:
                            score = base * g
                        seen.append(score)
                        if lam is None:
                            lam = float(np.median(seen))
                        target = max(1, N - RESERVE) * (((i + 1) / max(1, NSTEP - RESERVE)) ** GAMMA_PACE)
                        if spent < target - 0.5:
                            lam *= 0.85
                        elif spent > target + 0.5:
                            lam *= 1.15
                        do = score > lam
                elif op == 'tea_swap':
                    do = (i in Sset) or (not H)
                    if do and len(H) >= 2 and sm.n >= 2 and not DEFER[0] and (i + 1) < NSTEP - RESERVE:
                        s_, sp_ = H[-1], H[-2]
                        p_now = pred_gain(i, x, s_, sp_)
                        if p_now > 0:
                            ln = tok2lat(x)
                            ln = step(s_['v'], TS[i + 1], ln)
                            p_next = pred_gain(i + 1, lat2tok(ln), s_, sp_)
                            if p_next < 0.7 * p_now:
                                do = False
                                DEFER[0] = True
                    if DEFER[0] and not do:
                        DEFER[0] = False
                        do = True
                else:
                    do = (i in Sset) or (not H)

                if do:
                    chosen.append(i)
                    if op == 'tea_branch':
                        u_t, c_t = model_both(lat, pe, tt)
                        v = u_t + CFG * (c_t - u_t)
                        nfe += 1; spent += 1
                    else:
                        v = model(lat, pe, tt); nfe += 1; spent += 1
                    if op == 'mix' and len(H) >= 2:
                        e_, _, _ = experts(i, x)
                        g_ = (NSTEP - i) / NSTEP          # remaining-horizon weight: early errors propagate further
                        for k_ in W:
                            loss_ = g_ * float((e_[k_] - v).norm()) / (float(v.norm()) + 1e-12)
                            W[k_] *= math.exp(-ETA_MIX * min(loss_, 10.0))
                            W[k_] = max(W[k_], 1e-6)
                        z_ = sum(W.values())
                        for k_ in W:
                            W[k_] /= z_
                        if max(W.values()) < 1e-9:
                            W = {k_: 1.0 for k_ in W}
                    if len(H) >= 2:
                        k1, k2 = H[-1], H[-2]
                        if op == 'mp_reuse':
                            dvec = x - k1['x']
                            sm.observe(dvec, v - k1['v'], k1['v'], k1['v'],
                                       cap_base=float((k1['v'] - k2['v']).norm()))
                        else:
                            rt = (i - k1['i']) / (k1['i'] - k2['i'])
                            v_ex1_k = k1['v'] + rt * (k1['v'] - k2['v'])
                            dvec = (x - k1['x']) - rt * (k1['x'] - k2['x'])
                            if sm.innov_norm == 'r' and rt > 1e-6:
                                dvec = dvec / rt
                            sm.observe(dvec, v - v_ex1_k, v_ex1_k, k1['v'])
                    if op == 'tea_branch' and len(H) >= 2:
                        k1b, k2b = H[-1], H[-2]
                        rt_b = (i - k1b['i']) / (k1b['i'] - k2b['i'])
                        d_b = (x - k1b['x']) - rt_b * (k1b['x'] - k2b['x'])
                        u_ex1_k = k1b['u'] + rt_b * (k1b['u'] - k2b['u'])
                        c_ex1_k = k1b['c'] + rt_b * (k1b['c'] - k2b['c'])
                        sm_u.observe(d_b, u_t - u_ex1_k, u_ex1_k, k1b['u'])
                        sm_c.observe(d_b, c_t - c_ex1_k, c_ex1_k, k1b['c'])
                    ent = dict(i=i, x=x, v=v)
                    if op == 'tea_branch':
                        ent['u'], ent['c'] = u_t, c_t
                    H = (H + [ent])[-2:]
                else:
                    s = H[-1]
                    if len(H) < 2:
                        v = s['v']
                    else:
                        sp = H[-2]
                        rt = (i - s['i']) / (s['i'] - sp['i'])
                        v_ex1 = s['v'] + rt * (s['v'] - sp['v'])
                        if op == 'tea_branch':
                            if (i - s['i']) >= GATE_AGE and sm_u.n >= 2:
                                u_ex1 = s['u'] + rt * (s['u'] - sp['u'])
                                c_ex1 = s['c'] + rt * (s['c'] - sp['c'])
                                u_hat = sm_u.forecast(u_ex1, s['u'], dvec)
                                c_hat = sm_c.forecast(c_ex1, s['c'], dvec)
                                v = u_hat + CFG * (c_hat - u_hat)
                            else:
                                v = s['v']
                        elif op in ('tea_gap', 'tea_gap1'):
                            age_ = i - s['i']
                            if age_ >= GATE_AGE and sm.n >= 2:
                                f_ = sm.forecast(v_ex1, s['v'], dvec)
                                v = v_ex1 + TRUST * (f_ - v_ex1) if TRUST >= 0 else f_
                            else:
                                v = s['v']
                        elif op == 'tea_tok':
                            age_ = i - s['i']
                            if age_ >= GATE_AGE and sm.n >= 2:
                                f_ = sm.forecast(v_ex1, s['v'], dvec)
                                if len(H) >= 3 and TOK_Q >= 0:
                                    h1, h2, h3 = H[-1], H[-2], H[-3]
                                    dt1 = max(h1['i'] - h2['i'], 1)
                                    dt2 = max(h2['i'] - h3['i'], 1)
                                    v1 = (h1['x'] - h2['x']) / dt1
                                    v2 = (h2['x'] - h3['x']) / dt2
                                    acc = (v1 - v2) / max(dt1, 1)
                                    kap = acc.norm(dim=1) / (v1.norm(dim=1) ** 2 + 1e-6)
                                    thr = torch.quantile(kap, TOK_Q)
                                    m_ = (kap > thr).unsqueeze(1)
                                    v = torch.where(m_, f_, s['v'])
                                else:
                                    v = f_
                            else:
                                v = s['v']
                        elif op == 'tea_pgate':
                            s_, sp_ = H[-1], H[-2]
                            pr_ = pred_gain(i, x, s_, sp_) / (float(s_['v'].norm()) + 1e-12)
                            if pr_ > PGATE and sm.n >= 2:
                                v = sm.forecast(v_ex1, s['v'], dvec)
                            else:
                                v = s['v']
                        elif op == 'tea_mix':
                            age_ = i - s['i']
                            if age_ >= GATE_AGE and sm.n >= 2:
                                v = sm.forecast(v_ex1, s['v'], dvec)
                            elif age_ == 1 and sm.n >= 2:
                                f_ = sm.forecast(v_ex1, s['v'], dvec)
                                v = s['v'] + W1 * (f_ - s['v'])
                            else:
                                v = s['v']
                        elif op in ('tea_ex', 'tea_ex2'):
                            age_ = i - s['i']
                            prev_gap_ = s['i'] - sp['i']
                            if age_ >= GATE_AGE and sm.n >= 2:
                                v = sm.forecast(v_ex1, s['v'], dvec)
                            elif age_ == 1 and (op == 'tea_ex' or prev_gap_ == 1):
                                v = v_ex1
                            else:
                                v = s['v']
                        elif op == 'mix':
                            e_, _, _ = experts(i, x)
                            v = e_[max(W, key=W.get)]
                        elif op in ('reuse', 'mpc_reuse', 'mpc_in', 'mpc_tm', 'mpc_md', 'mpc_th_in', 'mpc_th_md', 'mp_reuse', 'peak', 'dp', 'tea_gap', 'tea_gap1', 'tea_ex', 'tea_ex2', 'tea_swap', 'tea_mix', 'tea_pgate'):
                            v = s['v']
                        elif op == 'ex1':
                            v = v_ex1
                        else:
                            dvec = (x - s['x']) - rt * (s['x'] - sp['x'])
                            if sm.innov_norm == 'r' and rt > 1e-6:
                                dvec = dvec / rt
                            v = sm.forecast(v_ex1, s['v'], dvec)
                lat = step(v, tt, lat)
            if op.startswith('tea_'):
                print(f'    [model] n={sm.n} beta={sm.beta:.3f} log_tail={[(round(a,2),round(b,2)) for a,b in sm.log[-3:]]}', flush=True)
            return lat, nfe, chosen

    for seed0 in seeds:
        for pi_ in range(a.nprompt):
            pi = pi_ + a.prompt_offset
            seed = pi if a.seed_from_prompt else seed0
            if all((pi, seed, N, op) in done for N in Ns for op in planners):
                continue
            pe = embs[pi]
            # The all-exact reference is expensive (NSTEP steps) and is reused by every
            # arm and every budget, so cache it per (prompt, seed) as fp32 on disk.
            zc = OUT / f'zfull_{pi}_{seed}.pt'
            if a.no_full:
                z_full = None
            elif zc.exists():
                z_full = torch.load(zc, map_location='cuda', weights_only=True).float()
            else:
                lat = init(seed)
                _tf = time.time()
                for tt in TS:
                    lat = step(model(lat, pe, tt), tt, lat)
                torch.cuda.synchronize()
                print(f'[full] {time.time() - _tf:.1f}s / {NSTEP} steps', flush=True)
                z_full = lat.float()
                torch.save(z_full.cpu(), zc)
            for N in Ns:
                for op in planners:
                    if (pi, seed, N, op) in done:
                        continue
                    lat, nfe, chosen = run_closed(pe, seed, N, op)
                    ok = bool(torch.isfinite(lat).all())
                    rr = rel(lat.float(), z_full) if (ok and z_full is not None) else float('nan')
                    if a.save_video and op in a.save_op.split(',') and N == a.save_n and ok:
                        save_video(lat, pi, op)
                    row = dict(prompt=pi, seed=seed, N=N, op=op, rel=rr, nfe=nfe, steps=chosen)
                    with open(LC, 'a') as f:
                        f.write(json.dumps(row) + '\n')
                    print(f'p{pi} s{seed} N={N:2d} {op:>7}: rel={rr:.4f} nfe={nfe} steps={chosen}', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()