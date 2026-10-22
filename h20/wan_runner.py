#!/usr/bin/env python3
"""Wan2.1-T2V-1.3B smoke test: does the stock diffusers pipeline load and run?

Stage 1 (this script): load the pipeline, run ONE prompt at small size, print time/VRAM.
If this passes, the caching runner can hook the transformer with confidence.

usage:
  python wan_smoke.py --frames 33 --height 256 --width 256 --steps 20
"""
import argparse, time, json
from pathlib import Path
import torch

ROOT = Path('/root/viewdit')
MODEL = ROOT / 'weights/Wan2.1-T2V-1.3B-Diffusers'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=str(MODEL))
    ap.add_argument('--frames', type=int, default=33)
    ap.add_argument('--height', type=int, default=256)
    ap.add_argument('--width', type=int, default=256)
    ap.add_argument('--steps', type=int, default=20)
    ap.add_argument('--prompt', default='A cat walking on a beach at sunset.')
    a = ap.parse_args()

    from diffusers import WanPipeline
    print('loading', a.model, flush=True)
    t0 = time.time()
    pipe = WanPipeline.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    pipe.to('cuda')
    pipe.set_progress_bar_config(disable=True)
    print(f'loaded in {time.time()-t0:.1f}s; '
          f'peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB', flush=True)

    torch.cuda.reset_peak_memory_stats()
    g = torch.Generator(device='cuda').manual_seed(3000)
    t0 = time.time()
    with torch.no_grad():
        out = pipe(prompt=a.prompt, height=a.height, width=a.width, num_frames=a.frames,
                   num_inference_steps=a.steps, guidance_scale=5.0, generator=g,
                   output_type='latent')
    dt = time.time() - t0
    z = out.frames if hasattr(out, 'frames') else out[0]
    print(json.dumps(dict(
        seconds=round(dt, 2),
        per_step=round(dt / a.steps, 3),
        shape=list(z.shape),
        dtype=str(z.dtype),
        finite=bool(torch.isfinite(z).all()),
        peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2),
    )), flush=True)


if __name__ == '__main__':
    main()
