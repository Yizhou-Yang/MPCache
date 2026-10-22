#!/usr/bin/env python3
"""F4/F5 of the oracle decomposition (CogVideoX-2b, DDIM-30, CFG 6, v-pred).

Every caching method = WHEN (which steps run the DiT) x WHAT (value used on skipped steps).
We replace each layer by an oracle to measure where the headroom is (Leis VLDB'15 analogy).

WHEN policies (all use the same WHAT unless stated; steps 0 and 29 always computed):
  uniform(N)     : N evenly spaced computed steps
  teacache(N)    : official TeaCache rule + official CogVideoX-2b coefficients. For CogVideoX the
                   indicator is the timestep embedding only -> the schedule is STATIC (prompt-
                   independent); threshold bisected so that exactly N steps are computed.
  easycache(tau) : EasyCache-lite, content adaptive: k = |dv|/|dx| at last computed pair,
                   accumulate k*|x_i-x_{i-1}|/|v_cache|, recompute when > tau.
  othr(tau)      : ORACLE signal: recompute when the TRUE error of the WHAT-prediction at this
                   step, |v_hat - v_full|/|v_full|, exceeds tau (same greedy-threshold policy,
                   true signal instead of a proxy -> cost of signal estimation error).
WHAT predictors (on skipped steps, from history of computed outputs):
  reuse : last computed v (FORA / TeaCache-style)
  ex1   : linear extrapolation from last 2 computed (TaylorSeer-1 at output level)
  ex2   : quadratic from last 3 computed
  lsq   : ORACLE hindsight-optimal linear combination of the last 3 computed v's, coefficients
          fitted by least squares to the true v_full at that step (upper bound on any linear
          multistep predictor whose basis is the history: Taylor/BDF/Adams/Hermite with <=3 pts)
Metric: final-latent relL2 vs full 30-step run; N = number of DiT calls used.
Oracle runs compute v_full at every step (accounting only counts used steps).
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path('/root/viewdit')
MODEL = ROOT / 'weights/CogVideoX-2b'
OUT = ROOT / 'results/video/step_oracle_20261015'
CFG = 6.0
NSTEP = 30
LT = 5
TEA_COEF = [-3.10658903e+01, 2.54732368e+01, -5.92380459e+00, 1.75769064e+00, -3.61568434e-03]

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


def rel(a, b):
    return float((a - b).norm() / (b.norm() + 1e-12))


def predict(what, hist, i, v_full=None):
    """hist: list of (step, v) computed, most recent last."""
    if what == 'reuse' or len(hist) == 1:
        return hist[-1][1]
    if what == 'ex1' or (what == 'ex2' and len(hist) < 3):
        (i0, v0), (i1, v1) = hist[-2:]
        return v1 + (v1 - v0) * (i - i1) / (i1 - i0)
    if what == 'ex2':
        (i0, v0), (i1, v1), (i2, v2) = hist[-3:]
        l0 = (i - i1) * (i - i2) / ((i0 - i1) * (i0 - i2))
        l1 = (i - i0) * (i - i2) / ((i1 - i0) * (i1 - i2))
        l2 = (i - i0) * (i - i1) / ((i2 - i0) * (i2 - i1))
        return l0 * v0 + l1 * v1 + l2 * v2
    if what == 'lsq':
        Hs = torch.stack([h.flatten() for _, h in hist[-3:]])          # [m, D]
        G = Hs @ Hs.T
        b = Hs @ v_full.flatten()
        a = torch.linalg.solve(G.double() + 1e-6 * torch.eye(len(G), device=G.device, dtype=torch.double) * G.diag().mean().double(),
                               b.double()).float()
        return (a[:, None] * Hs).sum(0).view_as(v_full)
    raise ValueError(what)


def main():
    from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
    from transformers import T5EncoderModel, T5Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='3000,3001')
    ap.add_argument('--nprompt', type=int, default=8)
    ap.add_argument('--Ns', default='10,13,17')
    ap.add_argument('--taus', default='0.04,0.07,0.11,0.16,0.24,0.35')
    ap.add_argument('--etaus', default='0.15,0.3,0.5,0.8,1.2,1.8')
    ap.add_argument('--only', default='')        # comma list of block names to run (debug)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    Ns = [int(x) for x in a.Ns.split(',')]
    taus = [float(x) for x in a.taus.split(',')]
    etaus = [float(x) for x in a.etaus.split(',')]
    only = set(a.only.split(',')) if a.only else None
    OUT.mkdir(parents=True, exist_ok=True)
    L = OUT / 'runs.jsonl'

    tok = T5Tokenizer.from_pretrained(MODEL / 'tokenizer')
    enc = T5EncoderModel.from_pretrained(MODEL / 'text_encoder', torch_dtype=torch.float16, attn_implementation='eager').cuda().eval()
    embs = {}
    with torch.no_grad():
        for pi in range(a.nprompt):
            ti = tok(PROMPTS[pi], padding='max_length', max_length=226, truncation=True, return_tensors='pt')
            embs[pi] = enc(ti.input_ids.cuda())[0].half()
        ne = enc(tok('', padding='max_length', max_length=226, truncation=True, return_tensors='pt').input_ids.cuda())[0].half()
    del enc; torch.cuda.empty_cache()

    dit = CogVideoXTransformer3DModel.from_pretrained(MODEL / 'transformer', torch_dtype=torch.float16).cuda().eval().requires_grad_(False)
    sch = CogVideoXDDIMScheduler.from_pretrained(MODEL / 'scheduler')
    sch.set_timesteps(NSTEP, device='cuda')
    TS = sch.timesteps

    # ---------- TeaCache static schedule (indicator = timestep embedding only) ----------
    with torch.no_grad():
        E = [dit.time_embedding(dit.time_proj(t.view(1)).to(torch.float16), None).float() for t in TS]
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
        lo, hi = 0., 50.
        best = None
        for _ in range(80):
            mid = (lo + hi) / 2; S = tea_sched(mid)
            if best is None or abs(len(S) - N) < abs(len(best) - N):
                best = S
            if len(S) > N: lo = mid
            elif len(S) < N: hi = mid
            else: return S
        return best

    tea = {N: sorted(tea_for(N)) for N in Ns}
    print('teacache dist', [round(d, 4) for d in dist[1:]], flush=True)
    for N in Ns:
        print(f'teacache N={N}: {tea[N]}', flush=True)

    @torch.no_grad()
    def model(lat, pe, t):
        out = dit(hidden_states=torch.cat([lat] * 2), encoder_hidden_states=torch.cat([ne, pe]),
                  timestep=t.expand(2), return_dict=False)[0].float()
        u, c = out.chunk(2)
        return u + CFG * (c - u)

    @torch.no_grad()
    def run(pe, seed, when, what='reuse', N=None, tau=None, need_full=False):
        g = torch.Generator(device='cuda').manual_seed(seed)
        lat = torch.randn((1, LT, 16, 60, 90), generator=g, device='cuda', dtype=torch.float16) * sch.init_noise_sigma
        if when == 'uniform':
            S = set(int(round(x)) for x in np.linspace(0, NSTEP - 1, N))
        elif when == 'teacache':
            S = set(tea[N])
        else:
            S = None
        hist, used, errs = [], [], []
        x_prev, k_rate, acc = None, None, 0.
        for i, t in enumerate(TS):
            forced = i in (0, NSTEP - 1)
            v_full = model(lat, pe, t) if (need_full or forced or (S is not None and i in S)) else None
            if when == 'full':
                compute = True
            elif S is not None:
                compute = i in S
            elif forced or len(hist) == 0:
                compute = True
            elif when == 'othr':
                v_hat = predict(what, hist, i, v_full)
                e = rel(v_hat, v_full)
                compute = e > tau
            elif when == 'easycache':
                if k_rate is None:
                    compute = True
                else:
                    acc += k_rate * float((lat.float() - x_prev).norm()) / float(hist[-1][1].norm())
                    compute = acc > tau
            if compute:
                if v_full is None:
                    v_full = model(lat, pe, t)
                v = v_full
                if when == 'easycache' and hist:
                    k_rate = float((v_full - hist[-1][1]).norm()) / (float((lat.float() - hist_x).norm()) + 1e-12)
                    acc = 0.
                hist = (hist + [(i, v_full)])[-3:]
                hist_x = lat.float()
                used.append(i)
            else:
                v = predict(what, hist, i, v_full) if what != 'lsq' else predict('lsq', hist, i, v_full)
                if v_full is not None:
                    errs.append((i, rel(v, v_full)))
            x_prev = lat.float()
            lat = sch.step(v.half(), t, lat, eta=0., return_dict=False)[0].half()
        assert torch.isfinite(lat).all()
        return lat.float(), used, errs

    def plan():
        P = []
        for N in Ns:
            for what in ('reuse', 'ex1', 'ex2'):
                P.append(('uniform', what, N, None, False))
            P.append(('uniform', 'lsq', N, None, True))
            P.append(('teacache', 'reuse', N, None, False))
            P.append(('teacache', 'ex1', N, None, False))
        for tau in etaus:
            P.append(('easycache', 'reuse', None, tau, False))
        for what in ('reuse', 'ex1'):
            for tau in taus:
                P.append(('othr', what, None, tau, True))
        if only:
            P = [p for p in P if p[0] in only or f'{p[0]}:{p[1]}' in only]
        return P

    P = plan()
    print(f'{len(P)} configs per sample', flush=True)
    for pi in range(a.nprompt):
        pe = embs[pi]
        for seed in seeds:
            t0 = time.time()
            z_full, _, _ = run(pe, seed, 'full')
            for when, what, N, tau, nf in P:
                t1 = time.time()
                z, used, errs = run(pe, seed, when, what, N, tau, nf)
                row = dict(prompt=pi, seed=seed, when=when, what=what, N_target=N, tau=tau,
                           N=len(used), rel=rel(z, z_full), used=used, errs=errs)
                with open(L, 'a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f'p{pi} s{seed} {when:>9}/{what:<5} N={len(used):2d} tau={tau} rel={row["rel"]:.4f} '
                      f'({time.time() - t1:.0f}s) used={used}', flush=True)
            print(f'  [p{pi} s{seed} {time.time() - t0:.0f}s]', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
