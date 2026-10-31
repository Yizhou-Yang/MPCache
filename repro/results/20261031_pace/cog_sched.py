import os, sys, torch, numpy as np, imageio.v2 as iio
from diffusers import CogVideoXPipeline
M = "/root/viewdit/weights/CogVideoX-2b"
NSTEP = 30

def patchify(t):
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
        self.num = torch.zeros(dim); self.den = torch.zeros(dim); self.n = 0
    def observe(self, d, y):
        self.num += (y * d).sum(dim=(0, 1)).cpu()
        self.den += (d * d).sum(dim=(0, 1)).cpu()
        self.n += 1
    def A(self):
        return (self.num / (self.den + 1e-8)).cuda()
    def drift(self, d, v_ref):
        A = self.A()
        return ((A * d).pow(2).mean().sqrt() / (v_ref.pow(2).mean().sqrt() + 1e-8)).item()

pipe = CogVideoXPipeline.from_pretrained(M, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.vae.enable_tiling()
tf = pipe.transformer; orig = tf.forward
prompts = open("/root/viewdit/resources/prompts/prompt_video.txt").read().splitlines()

def run(pi, tau, outdir, warmup=4, maxgap=4):
    st = {"i": -1, "h": [], "sm": Diag(), "nfe": 0}
    gap = {"g": 0}
    def fwd(*args, **kw):
        st["i"] += 1; i = st["i"]
        hs = kw.get("hidden_states", args[0] if args else None)
        xt = patchify(hs.detach().float())
        h = st["h"]
        do_eval = (i < warmup) or (i == NSTEP - 1) or (not h) or gap["g"] >= maxgap
        if not do_eval and st["sm"].n >= 2:
            (i0, x0, v0), (i1, x1, v1) = h[-2], h[-1]
            r = (i - i1) / max(1, (i1 - i0))
            d = (xt - x1) - r * (x1 - x0)
            do_eval = st["sm"].drift(d, v1) > tau
        if do_eval:
            out = orig(*args, **kw)
            o = out[0] if isinstance(out, (tuple, list)) else out
            vt = patchify(o.detach().float())
            if len(h) >= 2:
                (i0, x0, v0), (i1, x1, v1) = h[-2], h[-1]
                r = (i - i1) / max(1, (i1 - i0))
                v_ex1 = v1 + r * (v1 - v0)
                d = (xt - x1) - r * (x1 - x0)
                st["sm"].observe(d, vt - v_ex1)
            h.append((i, xt, vt))
            if len(h) > 2: h.pop(0)
            st["nfe"] += 1
            gap["g"] = 0
            return out
        gap["g"] += 1
        return (unpatchify(h[-1][2], hs.shape).to(hs.dtype),)
    tf.forward = fwd
    g = torch.Generator("cuda").manual_seed(pi)
    with torch.no_grad():
        out = pipe(prompt=prompts[pi], negative_prompt="animation", num_frames=17,
                   height=480, width=720, num_inference_steps=NSTEP, guidance_scale=6.0, generator=g)
    tf.forward = orig
    frames = np.stack([np.array(f).astype(np.uint8) for f in out.frames[0]])
    os.makedirs(outdir, exist_ok=True)
    iio.mimsave(os.path.join(outdir, "video_%06d.mp4" % pi), frames, fps=8, codec="libx264", macro_block_size=8)
    print("[done] p=%d tau=%.3f nfe=%d" % (pi, tau, st["nfe"]), flush=True)

for tau in (0.02, 0.05, 0.10):
    for pi in range(4):
        run(pi, tau, "results/cogsched/t%s" % str(tau).replace(".", ""))
print("SCHED_DONE", flush=True)
