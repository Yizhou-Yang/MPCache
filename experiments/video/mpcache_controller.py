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

FRAMES = {17: dict(lt=5, ph=30, pw=45, steps=30), 49: dict(lt=13, ph=60, pw=90, steps=50)}
MAX_GAP = 4
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

    def __init__(self, kind='diag'):
        self.kind = kind
        self.Sdd = torch.zeros(64, device='cuda', dtype=torch.float64)
        self.Sde = torch.zeros(64, device='cuda', dtype=torch.float64)
        self.G = torch.zeros(64, 64, device='cuda', dtype=torch.float64)
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
        return d * a.float()

    def forecast(self, v_ex1, v_s, d):
        corr = self.correction(d)
        cap = KAPPA * float((v_ex1 - v_s).norm())
        cn = float(corr.norm())
        if cn > cap:
            corr = corr * (cap / (cn + 1e-12))
        return v_ex1 + self.beta * corr

    def observe(self, d, e, v_ex1_k, v_prev):
        if self.n > 0:
            pred = self.forecast(v_ex1_k, v_prev, d)
            err_m, err_e = float((pred - (v_ex1_k + e)).norm()), float(e.norm())
            self.beta = min(1.0, self.beta + 0.25) if err_m < err_e else self.beta * 0.5
            self.log.append((err_m / (err_e + 1e-12), self.beta))
        dd, ee = d.double(), e.double()
        self.Sdd = GAMMA * self.Sdd + (dd * dd).sum(0)
        self.Sde = GAMMA * self.Sde + (dd * ee).sum(0)
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
    ap.add_argument('--Ns', default='10,13,17')
    ap.add_argument('--planners', default='reuse,ex1,tea_mp,mpc,mpc_reuse')
    ap.add_argument('--out', default=str(ROOT / 'results/video/mpc_20261020'))
    ap.add_argument('--max-gap', type=int, default=4, help='hard cap on consecutive skipped steps (mpc)')
    ap.add_argument('--sanity', action='store_true')
    a = ap.parse_args()

    global LT, PH, PW, NSTEP, MAX_GAP
    cfgf = FRAMES[a.frames]
    LT, PH, PW, NSTEP = cfgf['lt'], cfgf['ph'], cfgf['pw'], cfgf['steps']
    MAX_GAP = a.max_gap

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

    def run_closed(pe, seed, N, op):
        """One closed-loop run; returns terminal latent and nfe."""
        lat = init(seed)
        H, sm, nfe, spent = [], StateModel('diag'), 0, 0
        lam, seen = None, []
        Sset = set(tea[N]) if op != 'mpc' else None
        chosen = []
        for i, tt in enumerate(TS):
            x = lat2tok(lat)
            if op in ('mpc', 'mpc_reuse'):
                r_left, b_left = NSTEP - i, N - spent
                if b_left <= 0:
                    do = False
                elif b_left >= r_left:
                    do = True
                elif len(H) < 2 or sm.n < 2:
                    do = True                      # warm start: the model cannot score before it is fitted
                elif i - H[-1]['i'] >= MAX_GAP:
                    do = True                      # never extrapolate farther than MAX_GAP steps
                elif spent >= math.ceil(N * (i + 1) / NSTEP):
                    do = False                     # hard quota: never run ahead of an even spend
                else:
                    s, sp = H[-1], H[-2]
                    rt = (i - s['i']) / (s['i'] - sp['i'])
                    dvec = (x - s['x']) - rt * (s['x'] - sp['x'])
                    v_ex1 = s['v'] + rt * (s['v'] - sp['v'])
                    base = float((v_ex1 - s['v']).norm())
                    if op == 'mpc_reuse':
                        score = base * ((NSTEP - i) / NSTEP)          # pure temporal-change pricing
                    else:
                        corr = sm.forecast(v_ex1, s['v'], dvec) - v_ex1
                        score = (float(corr.norm()) if sm.n else base) * ((NSTEP - i) / NSTEP)
                    seen.append(score)
                    if lam is None:
                        lam = float(np.median(seen))
                    target = N * (i + 1) / NSTEP
                    if spent < target - 0.5:
                        lam *= 0.85
                    elif spent > target + 0.5:
                        lam *= 1.15
                    do = score > lam
            else:
                do = (i in Sset) or (not H)

            if do:
                chosen.append(i)
                v = model(lat, pe, tt); nfe += 1; spent += 1
                if len(H) >= 2:
                    k1, k2 = H[-1], H[-2]
                    rt = (i - k1['i']) / (k1['i'] - k2['i'])
                    v_ex1_k = k1['v'] + rt * (k1['v'] - k2['v'])
                    dvec = (x - k1['x']) - rt * (k1['x'] - k2['x'])
                    sm.observe(dvec, v - v_ex1_k, v_ex1_k, k1['v'])
                H = (H + [dict(i=i, x=x, v=v)])[-2:]
            else:
                s = H[-1]
                if len(H) < 2:
                    v = s['v']
                else:
                    sp = H[-2]
                    rt = (i - s['i']) / (s['i'] - sp['i'])
                    v_ex1 = s['v'] + rt * (s['v'] - sp['v'])
                    if op in ('reuse', 'mpc_reuse'):
                        v = s['v']
                    elif op == 'ex1':
                        v = v_ex1
                    else:
                        dvec = (x - s['x']) - rt * (s['x'] - sp['x'])
                        v = sm.forecast(v_ex1, s['v'], dvec)
            lat = step(v, tt, lat)
        return lat, nfe, chosen

    for seed in seeds:
        for pi in range(a.nprompt):
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
