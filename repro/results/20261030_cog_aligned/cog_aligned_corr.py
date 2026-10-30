import os, sys, torch, numpy as np, imageio.v2 as iio
from diffusers import CogVideoXPipeline

M = "/root/viewdit/weights/CogVideoX-2b"
NSTEP = 30

def sched(n_full, nstep=NSTEP, warmup=4):
    head = list(range(min(warmup, n_full)))
    rest = n_full - len(head) - 1
    if rest <= 0: return sorted(set(head + [nstep - 1]))
    span = nstep - 1 - head[-1]
    body = [head[-1] + int(round((k + 1) * span / (rest + 1))) for k in range(rest)]
    return sorted(set(head + body + [nstep - 1]))

def patchify(t):
    # CogVideoX uses [B, T, C, H, W]
    B, T, C, H, W = t.shape
    x = t.reshape(B, T, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(B, T * (H // 2) * (W // 2), C * 4)

def unpatchify(tok, shape):
    B, T, C, H, W = shape
    H2, W2 = H // 2, W // 2
    x = tok.reshape(B, T, H2, W2, C, 2, 2).permute(0, 1, 4, 2, 5, 3, 6)
    return x.reshape(B, T, C, H, W)

class Diag:
    def __init__(self, dim=64):
        self.num = torch.zeros(dim); self.den = torch.zeros(dim)
        self.beta = 0.5; self.n = 0
    def observe(self, d, y):
        self.num += (y * d).sum(dim=(0, 1)).cpu(); self.den += (d * d).sum(dim=(0, 1)).cpu()
        self.n += 1
        if self.n >= 2:
            A = (self.num / (self.den + 1e-8)).cuda()
            em = (y - A * d).pow(2).mean().sqrt().item()
            ee = y.pow(2).mean().sqrt().item() + 1e-8
            r = ee / (em + ee)
            self.beta = 0.7 * self.beta + 0.3 * r
    def correction(self, d):
        A = (self.num / (self.den + 1e-8)).cuda()
        return (A * d) * self.beta

pipe = CogVideoXPipeline.from_pretrained(M, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.vae.enable_tiling()
tf = pipe.transformer; orig = tf.forward
prompts = open("/root/viewdit/resources/prompts/prompt_video.txt").read().splitlines()

def run(pi, mode, outdir, nfull=19):
    full = sched(nfull) if nfull < NSTEP else list(range(NSTEP))
    fs, st = set(full), {"i": -1, "h": [], "sm": Diag()}
    def fwd(*args, **kw):
        st["i"] += 1; i = st["i"]
        hs = kw.get("hidden_states", args[0] if args else None)
        if i in fs or not st["h"]:
            out = orig(*args, **kw)
            o = out[0] if isinstance(out, (tuple, list)) else out
            xt = patchify(hs.detach().float())
            vt = patchify(o.detach().float())
            h = st["h"]
            if len(h) >= 2 and mode == "corr":
                (i0, x0, v0), (i1, x1, v1) = h[-2], h[-1]
                r = (i - i1) / max(1, (i1 - i0))
                v_ex1 = v1 + r * (v1 - v0)
                d = (xt - x1) - r * (x1 - x0)
                st["sm"].observe(d, vt - v_ex1)
            h.append((i, xt, vt))
            if len(h) > 2: h.pop(0)
            return out
        (i0, x0, v0), (i1, x1, v1) = st["h"][-2], st["h"][-1]
        r = (i - i1) / max(1, (i1 - i0))
        if mode in ("reuse",):
            vt = v1
        elif mode == "ex1":
            vt = v1 + r * (v1 - v0)
        else:
            v_ex1 = v1 + r * (v1 - v0)
            d = (patchify(hs.detach().float()) - x1) - r * (x1 - x0)
            vt = v_ex1 + st["sm"].correction(d) if st["sm"].n >= 2 else v1
        return (unpatchify(vt, hs.shape).to(hs.dtype),)
    tf.forward = fwd
    g = torch.Generator("cuda").manual_seed(pi)
    with torch.no_grad():
        out = pipe(prompt=prompts[pi], negative_prompt="animation", num_frames=17,
                   height=480, width=720, num_inference_steps=NSTEP, guidance_scale=6.0, generator=g)
    tf.forward = orig
    frames = np.stack([np.array(f).astype(np.uint8) for f in out.frames[0]])
    os.makedirs(outdir, exist_ok=True)
    iio.mimsave(os.path.join(outdir, "video_%06d.mp4" % pi), frames, fps=8, codec="libx264", macro_block_size=8)
    print("[done] p=%d mode=%s nfe=%d" % (pi, mode, len(full)), flush=True)

for pi in range(6):
    for mode in ("corr", "reuse"):
        run(pi, mode, "results/cogcorr/" + mode)
print("CORR_DONE", flush=True)
