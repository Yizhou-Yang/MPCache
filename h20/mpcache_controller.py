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
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/root/viewdit')
MODEL = ROOT / 'weights/CogVideoX-2b'
CFG = 6.0
C = 16
TEA_COEF = [-3.10658903e+01, 2.54732368e+01, -5.92380459e+00, 1.75769064e+00, -3.61568434e-03]
GAMMA, LAM_D, LAM_F, ACLIP, KAPPA, BETA0 = 0.5, 0.1, 0.1, 3.0, 1.0, 0.5
W1 = 0.3
PGATE = 0.02
BIAS = False
KAPPA_AGE = 0.0
TWOTAP = False
R_CLIP = 0.0
AUTO_BASE = False
RANK = 0
FORGET_ADAPT = False
import os as _os
_PROMPT_FILE = Path(_os.environ.get("MPCACHE_PROMPTS", "/root/viewdit/resources/prompts/prompt_video.txt"))

def _load_prompts():
    """Exact prompt list used by the caching baselines (RA-CFGCache / FasterCache set)."""
    if _PROMPT_FILE.exists():
        items = [l.strip() for l in _PROMPT_FILE.read_text().splitlines() if l.strip()]
        if items:
            return items
    return [
        'A white swan swims slowly across a calm blue pond, gentle ripples follow the swan, steady camera, realistic wildlife footage.',
        'A red bus drives slowly along a city street, buildings in the background, steady camera, realistic footage.',
        'A brown bear walks along a river bank in a dense forest, realistic wildlife footage.',
        'A panda eats bamboo while sitting in a lush green bamboo forest, close-up, realistic footage.',
        'Fireworks explode over a city skyline at night, reflections in the river, wide shot.',
        'A train crosses a tall bridge over a deep valley at sunrise, aerial view.',
        'A cat plays with a ball of yarn on a wooden floor, warm indoor lighting.',
        'A chef flips a pancake in a bright kitchen, slow motion.',
    ]

PROMPTS = _load_prompts()

FRAMES = {17: dict(lt=5, ph=30, pw=45, steps=30), 49: dict(lt=13, ph=60, pw=90, steps=50)}
MAX_GAP = 4
GAMMA_PACE = 1.0
RESERVE = 1
MODEL_KIND = 'diag'
GATE_AGE = 2
TRUST = -1.0
PEAK_LOOK = 6
PEAK_FRAC = 0.9
LT, PH, PW, NSTEP = 5, 30, 45, 30


def tok2lat(vt):
    o = vt.view(1, LT, PH, PW, C, 2, 2)
    return o.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)


def lat2tok(lat):
    return lat.float().view(1, LT, C, PH, 2, PW, 2).permute(0, 1, 3, 5, 2, 4, 6).reshape(LT * PH * PW, 64)


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
        self.Spp = torch.zeros(64, device='cuda', dtype=torch.float64)
        self.Spe = torch.zeros(64, device='cuda', dtype=torch.float64)
        self.Sdp = torch.zeros(64, device='cuda', dtype=torch.float64)
        self.d_prev = None
        self.alpha = None
        self.r_ema = None
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
            if RANK > 0:
                U, S_, Vh = torch.linalg.svd(A)
                A = (U[:, :RANK] * S_[:RANK]) @ Vh[:RANK, :]
            return (d.double() @ A).float()
        a = (self.Sde / ((1 + LAM_D) * self.Sdd + 1e-12)).clamp(-ACLIP, ACLIP)
        out = d * a.float()
        if TWOTAP and self.Spp is not None and float(self.Spp.sum()) > 1e-9:
            b = (self.Spe / ((1 + LAM_D) * self.Spp + 1e-12)).clamp(-ACLIP, ACLIP)
            if self.d_prev is not None:
                out = out + self.d_prev.float() * b.float()
        if BIAS and self.Sw > 1e-9:
            out = out + (self.Se / self.Sw).float().expand_as(out)
        return out

    def forecast(self, v_ex1, v_s, d, cap_base=None, age=1):
        corr = self.correction(d)
        base = cap_base if cap_base is not None else float((v_ex1 - v_s).norm())
        cap = KAPPA * base * (1.0 + KAPPA_AGE * max(0, age - 1))
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
        if FORGET_ADAPT:
            if self.alpha is None:
                self.alpha = torch.full((dd.shape[-1],), GAMMA, device=dd.device, dtype=torch.float64)
                self.r_ema = torch.zeros((dd.shape[-1],), device=dd.device, dtype=torch.float64)
            rc = (ee.abs() / (dd.abs() + 1e-6)).mean(0)
            self.r_ema = 0.8 * self.r_ema + 0.2 * rc
            ratio = (self.r_ema.median() + 1e-9) / (self.r_ema + 1e-9)
            self.alpha = (GAMMA * ratio).clamp(0.2, 0.9)
            gam = self.alpha
            if not getattr(self, '_dbg', False):
                self._dbg = True
                print('    [fadapt] alpha range %.3f..%.3f' % (float(gam.min()), float(gam.max())), flush=True)
        else:
            gam = GAMMA
        self.Sdd = gam * self.Sdd + (dd * dd).sum(0)
        self.Se = GAMMA * self.Se + ee.mean(0)
        if self.d_prev is not None and TWOTAP:
            dp = self.d_prev.double()
            self.Spp = GAMMA * self.Spp + (dp * dp).sum(0)
            self.Spe = GAMMA * self.Spe + (dp * ee).sum(0)
            self.Sdp = GAMMA * self.Sdp + (dd * dp).sum(0)
        self.d_prev = dd.clone()
        self.Sw = GAMMA * self.Sw + 1.0
        self.Sde = gam * self.Sde + (dd * ee).sum(0)
        self.G = GAMMA * self.G + dd.T @ dd
        self.R = GAMMA * self.R + dd.T @ ee
        self.n += 1


def main():
    from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
    from transformers import T5EncoderModel, T5Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=17, choices=[17, 49])
    ap.add_argument('--seeds', default='3000,3001')
    ap.add_argument('--nprompt', type=int, default=8)
    ap.add_argument('--prompt-offset', type=int, default=0)
    ap.add_argument('--seed-from-prompt', action='store_true',
                    help='use seed == global prompt index (RA-CFGCache semantics of --seed 0)')
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
    ap.add_argument('--kappa-age', type=float, default=0.0)
    ap.add_argument('--twotap', type=int, default=0)
    ap.add_argument('--r-clip', type=float, default=0.0)
    ap.add_argument('--rank', type=int, default=0, help='truncate the fitted operator to this rank (0 = off)')
    ap.add_argument('--forget-adapt', type=int, default=0, help='per-channel forget rate from the fit residual')
    ap.add_argument('--print-config', action='store_true', help='print the effective value of every tuning global')
    ap.add_argument('--auto-base', type=int, default=0,
                    help='on a uniform schedule use pure temporal extrapolation (provably right for a linear field)')
    ap.add_argument('--pgate', type=float, default=0.02)
    ap.add_argument('--gamma', type=float, default=0.6, help='budget pacing exponent: quota_i = N*((i+1)/T)**gamma')
    ap.add_argument('--sanity', action='store_true')
    a = ap.parse_args()

    global LT, PH, PW, NSTEP, MAX_GAP, GAMMA_PACE, RESERVE
    cfgf = FRAMES[a.frames]
    LT, PH, PW, NSTEP = cfgf['lt'], cfgf['ph'], cfgf['pw'], cfgf['steps']
    MAX_GAP = a.max_gap
    GAMMA_PACE = a.gamma
    global MODEL_KIND, GATE_AGE, TRUST, KAPPA, W1, GAMMA, BIAS, KAPPA_AGE, TWOTAP, R_CLIP, PGATE, RANK, AUTO_BASE, FORGET_ADAPT
    MODEL_KIND = a.model_kind
    GATE_AGE = a.gate_age
    TRUST = a.trust
    KAPPA = a.kappa
    W1 = a.w1
    GAMMA = a.rls_gamma
    BIAS = bool(a.bias)
    KAPPA_AGE = a.kappa_age
    TWOTAP = bool(a.twotap)
    R_CLIP = a.r_clip
    AUTO_BASE = bool(a.auto_base)
    RANK = a.rank
    FORGET_ADAPT = bool(a.forget_adapt)
    PGATE = a.pgate
    RESERVE = a.reserve_last
    if a.print_config:
        print('[config] gate_age=%s trust=%s kappa=%s kappa_age=%s gamma=%s bias=%s '
              'twotap=%s r_clip=%s rank=%s forget_adapt=%s auto_base=%s model=%s'
              % (GATE_AGE, TRUST, KAPPA, KAPPA_AGE, GAMMA, BIAS, TWOTAP, R_CLIP, RANK,
                 FORGET_ADAPT, AUTO_BASE, MODEL_KIND), flush=True)

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
    if emb_path.exists():
        cache = torch.load(emb_path, map_location='cuda', weights_only=True)
        embs, ne = {int(k): v for k, v in cache['embs'].items()}, cache['ne']
    else:
        tok = T5Tokenizer.from_pretrained(MODEL / 'tokenizer')
        enc = T5EncoderModel.from_pretrained(MODEL / 'text_encoder', torch_dtype=torch.float16,
                                             attn_implementation='eager').cuda().eval()
        embs = {}
        with torch.no_grad():
            for pi in range(len(PROMPTS)):
                ti = tok(PROMPTS[pi], padding='max_length', max_length=226, truncation=True, return_tensors='pt')
                embs[pi] = enc(ti.input_ids.cuda())[0].half()
            ne = enc(tok('', padding='max_length', max_length=226, truncation=True,
                         return_tensors='pt').input_ids.cuda())[0].half()
        del enc; torch.cuda.empty_cache()
        torch.save({'embs': {str(k): v for k, v in embs.items()}, 'ne': ne}, emb_path)

    d = CogVideoXTransformer3DModel.from_pretrained(MODEL / 'transformer', torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
    sch = CogVideoXDDIMScheduler.from_pretrained(MODEL / 'scheduler')
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
        return sch.step(tok2lat(v).half(), t, lat, eta=0., return_dict=False)[0].half()

    def init(seed):
        sch.set_timesteps(NSTEP, device='cuda')
        g = torch.Generator(device='cuda').manual_seed(seed)
        return torch.randn((1, LT, C, PH * 2, PW * 2), generator=g, device='cuda', dtype=torch.float16) * sch.init_noise_sigma

    lat = init(0)
    print('sanity roundtrip', rel(tok2lat(lat2tok(lat)), lat.float()), flush=True)
    torch.cuda.synchronize(); t0 = time.time(); model(lat, embs[0], TS[min(5, NSTEP - 1)]); torch.cuda.synchronize()
    print(f'one CFG step {time.time() - t0:.2f}s; peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)
    if a.sanity:
        return

    with torch.no_grad():
        E = [d.time_embedding(d.time_proj(tt.view(1)).to(torch.float16), None).float() for tt in TS]
    poly = np.poly1d(TEA_COEF)
    dist = [None] + [float(poly(((E[i] - E[i - 1]).abs().mean() / E[i - 1].abs().mean()).item())) for i in range(1, NSTEP)]

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
            tok_hist = []
            pt_err = None
            pt_prev = None
            ema3 = [None, None, None]   # reuse, ex1, model
            sm_c = StateModel('diag')
            lam, seen = None, []
            acc, theta, ema = 0.0, None, None
            PLAN = None
            W = {'reuse': 1.0, 'ex1': 1.0, 'model': 1.0}
            ETA_MIX = 1.0
            Sset = set(tea[N]) if op not in ('mpc', 'mpc_th_in', 'mpc_th_md', 'peak', 'dp') else None
            UNIFORM_SCHED = False
            if AUTO_BASE and Sset:
                _gs = sorted(Sset)
                _gaps = [_gs[j] - _gs[j-1] for j in range(1, len(_gs))]
                UNIFORM_SCHED = (len(set(_gaps)) == 1)
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
                            rt_eff = min(rt, R_CLIP) if R_CLIP > 0 else rt
                            v_ex1 = s['v'] + rt_eff * (s['v'] - sp['v'])
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
                        rt_eff = min(rt, R_CLIP) if R_CLIP > 0 else rt
                        v_ex1 = s['v'] + rt_eff * (s['v'] - sp['v'])
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
                    if op in ('tea_auto', 'tea_ratio') and len(H) >= 2:
                        k1a, k2a = H[-1], H[-2]
                        rt_a = (i - k1a['i']) / (k1a['i'] - k2a['i'])
                        v1_, v2_ = k1a['v'], k2a['v']
                        ex_ = v1_ + rt_a * (v1_ - v2_)
                        d_a = (x - k1a['x']) - rt_a * (k1a['x'] - k2a['x'])
                        fm_ = sm.forecast(ex_, v1_, d_a) if sm.n >= 2 else ex_
                        for j, cand in enumerate((v1_, ex_, fm_)):
                            e_ = float((v - cand).norm()) / (float(v.norm()) + 1e-12)
                            ema3[j] = e_ if ema3[j] is None else 0.7 * ema3[j] + 0.3 * e_
                    if op == 'tea_pt' and pt_prev is not None:
                        e_i = (v - pt_prev).norm(dim=-1, keepdim=True)
                        pt_err = e_i if pt_err is None else 0.7 * pt_err + 0.3 * e_i
                        pt_prev = None
                    if op in ('tea_tok', 'tea_curv'):
                        tok_hist.append(v.detach().float().clone())
                        if len(tok_hist) > 3:
                            tok_hist.pop(0)
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
                        rt_eff = min(rt, R_CLIP) if R_CLIP > 0 else rt
                        v_ex1 = s['v'] + rt_eff * (s['v'] - sp['v'])
                        if op == 'tea_ratio':
                            if sm.n >= 2 and all(e is not None for e in ema3):
                                f_ = sm.forecast(v_ex1, s['v'], dvec)
                                beta_dyn = max(0.0, min(1.0, 1.0 - ema3[2] / (ema3[1] + 1e-12)))
                                v = v_ex1 + beta_dyn * (f_ - v_ex1)
                            else:
                                v = v_ex1 if sm.n >= 2 else s['v']
                        elif op == 'tea_auto':
                            if (i - s['i']) >= 1 and sm.n >= 2 and all(e is not None for e in ema3):
                                f_ = sm.forecast(v_ex1, s['v'], dvec)
                                cands = (s['v'], v_ex1, f_)
                                v = cands[int(min(range(3), key=lambda j: ema3[j]))]
                            else:
                                v = v_ex1 if sm.n >= 2 else s['v']
                        elif op == 'tea_pt':
                            if (i - s['i']) >= GATE_AGE and sm.n >= 2 and pt_err is not None:
                                f_ = sm.forecast(v_ex1, s['v'], dvec)
                                hot = (pt_err > pt_err.median()).expand_as(f_)
                                v = torch.where(hot, f_, s['v'])
                                pt_prev = v.detach().clone()
                            else:
                                v = s['v']
                                pt_prev = v.detach().clone()
                        elif op == 'tea_curv':
                            if (i - s['i']) >= GATE_AGE and sm.n >= 2 and len(tok_hist) >= 3:
                                f_ = sm.forecast(v_ex1, s['v'], dvec)
                                y1, y2, y3 = tok_hist[-1], tok_hist[-2], tok_hist[-3]
                                vel = (y1 - y2)
                                acc = (y1 - 2 * y2 + y3)
                                kap = acc.norm(dim=-1) / (vel.norm(dim=-1) ** 2 + 1e-6)
                                thr = kap.median()
                                hot = (kap > thr).unsqueeze(-1)
                                # stable tokens: reuse; chaotic tokens: damped mix of
                                # extrapolation and model correction (WorldCache-style)
                                damp = 0.5 * v_ex1 + 0.5 * f_
                                v = torch.where(hot, damp, s['v'])
                            else:
                                v = s['v']
                        elif op == 'tea_tok':
                            if (i - s['i']) >= GATE_AGE and sm.n >= 2 and len(tok_hist) >= 3:
                                f_ = sm.forecast(v_ex1, s['v'], dvec)
                                y1, y2, y3 = tok_hist[-1], tok_hist[-2], tok_hist[-3]
                                vel = (y1 - y2)
                                acc = (y1 - 2 * y2 + y3)
                                kap = acc.norm(dim=-1) / (vel.norm(dim=-1) ** 2 + 1e-6)
                                thr = kap.median()
                                mask = (kap > thr).unsqueeze(-1)
                                v = torch.where(mask, f_, s['v'])
                            else:
                                v = s['v']
                        elif op == 'tea_branch':
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
                            if AUTO_BASE and UNIFORM_SCHED:
                                v = v_ex1 if sm.n >= 2 else s['v']
                            elif age_ >= GATE_AGE and sm.n >= 2:
                                f_ = sm.forecast(v_ex1, s['v'], dvec, age=age_)
                                v = v_ex1 + TRUST * (f_ - v_ex1) if TRUST >= 0 else f_
                            else:
                                v = s['v']
                        elif op == 'tea_pgate':
                            s_, sp_ = H[-1], H[-2]
                            pr_ = pred_gain(i, x, s_, sp_) / (float(s_['v'].norm()) + 1e-12)
                            if pr_ > PGATE and sm.n >= 2:
                                v = sm.forecast(v_ex1, s['v'], dvec, age=age_)
                            else:
                                v = s['v']
                        elif op == 'tea_mix':
                            age_ = i - s['i']
                            if age_ >= GATE_AGE and sm.n >= 2:
                                v = sm.forecast(v_ex1, s['v'], dvec, age=age_)
                            elif age_ == 1 and sm.n >= 2:
                                f_ = sm.forecast(v_ex1, s['v'], dvec)
                                v = s['v'] + W1 * (f_ - s['v'])
                            else:
                                v = s['v']
                        elif op in ('tea_ex', 'tea_ex2'):
                            age_ = i - s['i']
                            prev_gap_ = s['i'] - sp['i']
                            if age_ >= GATE_AGE and sm.n >= 2:
                                v = sm.forecast(v_ex1, s['v'], dvec, age=age_)
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

    for seed in (seeds[:1] if a.seed_from_prompt else seeds):
        for pi_ in range(a.nprompt):
            pi = pi_ + a.prompt_offset
            if a.seed_from_prompt:
                seed = pi
            if all((pi, seed, N, op) in done for N in Ns for op in planners):
                continue
            pe = embs[pi]
            lat = init(seed)
            for tt in TS:
                lat = step(model(lat, pe, tt), tt, lat)
            z_full = lat.float()
            for N in Ns:
                for op in planners:
                    if (pi, seed, N, op) in done:
                        continue
                    lat, nfe, chosen = run_closed(pe, seed, N, op)
                    ok = bool(torch.isfinite(lat).all())
                    rr = rel(lat.float(), z_full) if ok else float('nan')
                    row = dict(prompt=pi, seed=seed, N=N, op=op, rel=rr, nfe=nfe, steps=chosen)
                    with open(LC, 'a') as f:
                        f.write(json.dumps(row) + '\n')
                    print(f'p{pi} s{seed} N={N:2d} {op:>7}: rel={rr:.4f} nfe={nfe} steps={chosen}', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
