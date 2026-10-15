#!/usr/bin/env python3
"""G0a: cost-model fidelity for MPCache (CogVideoX-2b, DDIM-30, CFG 6, v-pred).

The axis of the paper: caching poses two questions (what to output when skipping, and
whether skipping is worth it). Existing methods answer the second by looking at the LOCAL
error of the forecast (greedy threshold). We claim the right cost is the PROPAGATED terminal
error, which a local linear model of the denoiser v ~= v_s + J(x - x_s) can price.

This experiment tests that claim directly: does a propagation-aware cost (built from an
online-identified local Jacobian J) predict the TRUE terminal error of a single skipped step
much better than the greedy local error does?

Per (prompt, seed):
 1. reference full 30-step run -> store x_i (latent before step i), v_i, z_full.
 2. identify J online, 1 extra DiT call/step, directional finite difference at fixed t:
        d_i = normalize(x_i - x_{i-1});  v_plus = model(x_i + eps*d_i, t_i)
        lam_i      = <v_plus - v_i, d_i> / eps                  (scalar contraction)
        lam_i[c]   = <(v_plus-v_i)[c], d_i[c]> / (eps ||d_i[c]||^2)   (per-channel, [16])
 3. for each skip step i (subset): reuse-forecast error dv_i = v_{i-1} - v_i
        L_i  = ||dv_i|| / ||v_i||             greedy local error (TeaCache-style signal)
        C0_i = ||tangent(dv_i, J=0)|| / ||z_full||   solver-only propagation (no denoiser amp)
        C_i  = ||tangent(dv_i, J=per-channel)|| / ||z_full||  model propagation
    tangent(dv_i) [fp32, exact affine DDIM step]:
        dx = step(v_i+dv_i) - step(v_i)                    # delta x_{i+1}
        for j in i+1..N-1:  dv_j = J_j(dx);  dx = step(v_j+dv_j, x_j+dx) - step(v_j, x_j)
 4. true terminal error T_i via suffix re-run (same fp16 loop as reference):
        start x_i, forecast vhat = v_{i-1}, then compute i+1..N-1 fully -> z_skip; T_i=rel(z_skip,z_full)

Gate (16 samples): Spearman(C, T) >= 0.7 and clearly > Spearman(L, T). Also report Spearman(C0, T)
to isolate how much the denoiser Jacobian adds over the bare solver.
"""
import argparse, json, time
from pathlib import Path
import torch

ROOT = Path('/root/viewdit')
MODEL = ROOT / 'weights/CogVideoX-2b'
OUT = ROOT / 'results/video/g0a_costmodel_20261015'
CFG, NSTEP, LT = 6.0, 30, 5
from step_oracle import PROMPTS, rel   # noqa: E402


def main():
    from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
    from transformers import T5EncoderModel, T5Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='3000,3001')
    ap.add_argument('--nprompt', type=int, default=8)
    ap.add_argument('--skips', default='1,3,5,7,9,11,13,15,17,19,21,23')
    ap.add_argument('--eps', type=float, default=0.02)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    skips = [int(s) for s in a.skips.split(',')]
    OUT.mkdir(parents=True, exist_ok=True)
    L = OUT / 'runs.jsonl'

    tok = T5Tokenizer.from_pretrained(MODEL / 'tokenizer')
    enc = T5EncoderModel.from_pretrained(MODEL / 'text_encoder', torch_dtype=torch.float16,
                                         attn_implementation='eager').cuda().eval()
    embs = {}
    with torch.no_grad():
        for pi in range(a.nprompt):
            ti = tok(PROMPTS[pi], padding='max_length', max_length=226, truncation=True, return_tensors='pt')
            embs[pi] = enc(ti.input_ids.cuda())[0].half()
        ne = enc(tok('', padding='max_length', max_length=226, truncation=True,
                     return_tensors='pt').input_ids.cuda())[0].half()
    del enc; torch.cuda.empty_cache()

    dit = CogVideoXTransformer3DModel.from_pretrained(MODEL / 'transformer', torch_dtype=torch.float16
                                                      ).cuda().eval().requires_grad_(False)
    sch = CogVideoXDDIMScheduler.from_pretrained(MODEL / 'scheduler')
    sch.set_timesteps(NSTEP, device='cuda')
    TS = sch.timesteps

    @torch.no_grad()
    def model(lat, pe, t):
        out = dit(hidden_states=torch.cat([lat] * 2), encoder_hidden_states=torch.cat([ne, pe]),
                  timestep=t.expand(2), return_dict=False)[0].float()
        u, c = out.chunk(2)
        return u + CFG * (c - u)

    def step32(v, t, x):
        return sch.step(v, t, x, eta=0., return_dict=False)[0]

    for pi in range(a.nprompt):
        pe = embs[pi]
        for seed in seeds:
            t0 = time.time()
            g = torch.Generator(device='cuda').manual_seed(seed)
            lat = torch.randn((1, LT, 16, 60, 90), generator=g, device='cuda', dtype=torch.float16) * sch.init_noise_sigma
            xs, vs = [], []
            for i, t in enumerate(TS):
                xs.append(lat.float().clone())
                v = model(lat, pe, t)
                vs.append(v.clone())
                lat = sch.step(v.half(), t, lat, eta=0., return_dict=False)[0].half()
            z_full = lat.float()
            N = NSTEP

            # ---- online J: directional finite difference at fixed t (1 extra DiT call/step) ----
            lam = [None] * N          # scalar
            lamc = [None] * N         # per-channel [16]
            for i in range(1, N):
                d = xs[i] - xs[i - 1]
                dn = d / (d.norm() + 1e-12)
                vp = model((xs[i] + a.eps * dn).half(), pe, TS[i])
                dv = vp - vs[i]
                lam[i] = float((dv * dn).sum()) / a.eps
                num = (dv * dn).sum(dim=(0, 1, 3, 4))          # [16]
                den = (dn * dn).sum(dim=(0, 1, 3, 4))          # [16]
                lamc[i] = (num / (a.eps * den + 1e-12)).float()

            def tangent(i, dv, mode):
                x, v = xs[i], vs[i]
                dx = step32(v + dv, TS[i], x) - step32(v, TS[i], x)
                for j in range(i + 1, N):
                    if mode == 'none':
                        dvv = torch.zeros_like(dx)
                    elif mode == 'scalar':
                        dvv = lam[j] * dx
                    else:                                       # channel
                        dvv = lamc[j].view(1, 1, 16, 1, 1) * dx
                    dx = step32(vs[j] + dvv, TS[j], xs[j] + dx) - step32(vs[j], TS[j], xs[j])
                return dx

            zn = z_full.norm() + 1e-12
            for i in skips:
                dv = vs[i - 1] - vs[i]
                # NOTE (fix 2026-10-15): these are norm RATIOS ||a||/||b||, not rel(a,b)=||a-b||/||b||;
                # the first launch used rel() here, which returned ~1 for every C and was discarded.
                L_i = float(dv.norm() / (vs[i].norm() + 1e-12))
                d0, ds, dc = tangent(i, dv, 'none'), tangent(i, dv, 'scalar'), tangent(i, dv, 'channel')
                C0, CS, C = float(d0.norm() / zn), float(ds.norm() / zn), float(dc.norm() / zn)
                # true terminal error: suffix re-run in the SAME fp16 loop as the reference
                z = xs[i].half().clone()
                z = sch.step(vs[i - 1].half(), TS[i], z, eta=0., return_dict=False)[0].half()
                for j in range(i + 1, N):
                    vv = model(z, pe, TS[j])
                    z = sch.step(vv.half(), TS[j], z, eta=0., return_dict=False)[0].half()
                dz = z.float() - z_full
                T_i = float(dz.norm() / zn)
                # direction fidelity of the linearised prediction (does J predict WHAT goes wrong?)
                cos0 = float((d0 * dz).sum() / (d0.norm() * dz.norm() + 1e-12))
                cosc = float((dc * dz).sum() / (dc.norm() * dz.norm() + 1e-12))
                errc = float((dc - dz).norm() / (dz.norm() + 1e-12))
                with open(L, 'a') as f:
                    f.write(json.dumps(dict(prompt=pi, seed=seed, i=i, local=L_i, c0=C0, cs=CS, c=C, true=T_i,
                                            cos0=cos0, cosc=cosc, errc=errc)) + '\n')
                print(f'p{pi} s{seed} i={i:2d} local={L_i:.4f} c0={C0:.4f} cs={CS:.4f} c={C:.4f} '
                      f'true={T_i:.4f} cos0={cos0:.3f} cosc={cosc:.3f}', flush=True)
            print(f'  [p{pi} s{seed} {time.time() - t0:.0f}s]', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
