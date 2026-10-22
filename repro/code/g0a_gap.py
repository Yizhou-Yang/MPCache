#!/usr/bin/env python3
"""Gap experiment: does the local signal still rank skips at larger gaps?

For an anchor step i and a gap g, steps i..i+g-1 are filled with the same stale value
vs[i-1]; steps i+g.. are computed exactly. Scores:
  local_end  ||vs[i-1] - vs[i+g-1]|| / ||vs[i+g-1]||
  local_sum  sum_k ||vs[i-1] - vs[i+k]|| / ||vs[i+k]||
  prior      solver-only tangent propagation of the injection at i
  online     channel-model tangent propagation of the injection at i
  true       measured terminal error of the gap
"""
import argparse, json, time
from pathlib import Path
import torch

ROOT = Path('/root/viewdit')
MODEL = ROOT / 'weights/CogVideoX-2b'
OUT = ROOT / 'results/video/g0a_gap_20261020'
CFG, NSTEP, LT = 6.0, 30, 5
from step_oracle import PROMPTS, rel   # noqa: E402


def main():
    from diffusers import CogVideoXTransformer3DModel, CogVideoXDDIMScheduler
    from transformers import T5EncoderModel, T5Tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='3000')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--nprompt', type=int, default=2)
    ap.add_argument('--skips', default='3,7,11,15,19')
    ap.add_argument('--gaps', default='1,2,3,5')
    ap.add_argument('--eps', type=float, default=0.02)
    a = ap.parse_args()
    seeds = [int(s) for s in a.seeds.split(',')]
    skips = [int(s) for s in a.skips.split(',')]
    gaps = [int(s) for s in a.gaps.split(',')]
    OUT.mkdir(parents=True, exist_ok=True)
    L = OUT / 'runs.jsonl'

    tok = T5Tokenizer.from_pretrained(MODEL / 'tokenizer')
    enc = T5EncoderModel.from_pretrained(MODEL / 'text_encoder', torch_dtype=torch.float16,
                                         attn_implementation='eager').cuda().eval()
    embs = {}
    with torch.no_grad():
        for pi in range(a.nprompt):
            ti = tok(PROMPTS[pi + a.offset], padding='max_length', max_length=226, truncation=True, return_tensors='pt')
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

            lamc = [None] * N
            for i in range(1, N):
                d = xs[i] - xs[i - 1]
                dn = d / (d.norm() + 1e-12)
                vp = model((xs[i] + a.eps * dn).half(), pe, TS[i])
                dv = vp - vs[i]
                num = (dv * dn).sum(dim=(0, 1, 3, 4))
                den = (dn * dn).sum(dim=(0, 1, 3, 4))
                lamc[i] = (num / (a.eps * den + 1e-12)).float()

            def tangent(i, gap, stale, mode):
                """Propagate the stale-output error through the skipped window, then the suffix."""
                dx = torch.zeros_like(xs[i])
                for j in range(i, i + gap):
                    dv = (stale - vs[j]) + dx            # stale output plus the state error so far
                    dx = step32(vs[j] + dv, TS[j], xs[j] + dx) - step32(vs[j], TS[j], xs[j])
                for j in range(i + gap, N):
                    dvv = torch.zeros_like(dx) if mode == 'none' else lamc[j].view(1, 1, 16, 1, 1) * dx
                    dx = step32(vs[j] + dvv, TS[j], xs[j] + dx) - step32(vs[j], TS[j], xs[j])
                return dx

            zn = z_full.norm() + 1e-12
            for i in skips:
                for gap in gaps:
                    if i + gap >= N:
                        continue
                    stale = vs[i - 1]
                    local_end = float((stale - vs[i + gap - 1]).norm() / (vs[i + gap - 1].norm() + 1e-12))
                    # implementable proxy: input change over the gap (no denoiser output needed)
                    proxy_end = float((xs[i + gap - 1] - xs[i - 1]).norm() / (xs[i - 1].norm() + 1e-12))
                    proxy_sum = sum(float((xs[i + k] - xs[i - 1]).norm() / (xs[i - 1].norm() + 1e-12))
                                    for k in range(gap))
                    local_sum = sum(float((stale - vs[i + k]).norm() / (vs[i + k].norm() + 1e-12))
                                    for k in range(gap))
                    dv = stale - vs[i]
                    C0 = float(tangent(i, gap, stale, 'none').norm() / zn)
                    C = float(tangent(i, gap, stale, 'channel').norm() / zn)
                    z = xs[i].half().clone()
                    for j in range(i, i + gap):
                        z = sch.step(stale.half(), TS[j], z, eta=0., return_dict=False)[0].half()
                    for j in range(i + gap, N):
                        vv = model(z, pe, TS[j])
                        z = sch.step(vv.half(), TS[j], z, eta=0., return_dict=False)[0].half()
                    T = float((z.float() - z_full).norm() / zn)
                    with open(L, 'a') as f:
                        f.write(json.dumps(dict(prompt=pi, seed=seed, i=i, gap=gap, local_end=local_end,
                                                local_sum=local_sum, proxy_end=proxy_end, proxy_sum=proxy_sum,
                                                prior=C0, online=C, true=T)) + '\n')
                    print(f'p{pi} s{seed} i={i:2d} g={gap} local={local_end:.4f} prior={C0:.4f} '
                          f'online={C:.4f} true={T:.4f}', flush=True)
            print(f'  [p{pi} s{seed} {time.time() - t0:.0f}s]', flush=True)
    print('[done]', flush=True)


if __name__ == '__main__':
    main()
