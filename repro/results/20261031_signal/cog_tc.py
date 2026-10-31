import os, sys, torch, numpy as np, imageio.v2 as iio
from diffusers import CogVideoXPipeline
M = "/root/viewdit/weights/CogVideoX-2b"; NSTEP = 30

def patchify(t):
    B, T, C, H, W = t.shape
    x = t.reshape(B, T, C, H // 2, 2, W // 2, 2).permute(0, 1, 3, 5, 2, 4, 6)
    return x.reshape(B, T * (H // 2) * (W // 2), C * 4)

pipe = CogVideoXPipeline.from_pretrained(M, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True); pipe.vae.enable_tiling()
tf = pipe.transformer; orig = tf.forward
prompts = open("/root/viewdit/resources/prompts/prompt_video.txt").read().splitlines()

def run(pi, tau, outdir):
    st = {"i": -1, "prev": None, "last": None, "acc": 0.0, "nfe": 0}
    def fwd(*args, **kw):
        st["i"] += 1; i = st["i"]
        hs = kw.get("hidden_states", args[0] if args else None)
        cur = hs.detach().float()
        if st["prev"] is None:
            do = True
        else:
            rel = (cur - st["prev"]).abs().mean().item() / (st["prev"].abs().mean().item() + 1e-8)
            st["acc"] += rel
            do = st["acc"] > tau or st["last"] is None or i == NSTEP - 1
        st["prev"] = cur
        if do:
            out = orig(*args, **kw)
            o = out[0] if isinstance(out, (tuple, list)) else out
            st["last"] = o.detach(); st["nfe"] += 1; st["acc"] = 0.0
            return out
        return (st["last"],)
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

if sys.argv[1] == "cal":
    for tau in (0.01, 0.02, 0.04, 0.08):
        run(0, tau, "results/cogtc/cal_%s" % str(tau).replace(".", ""))
else:
    tau = float(sys.argv[1])
    for pi in range(20):
        run(pi, tau, "results/cogtc/t%s" % str(tau).replace(".", ""))
print("TC_DONE", flush=True)
