import os, torch, numpy as np, imageio.v2 as iio
from diffusers import CogVideoXPipeline
M = "/root/viewdit/weights/CogVideoX-2b"; NSTEP = 30

def sched(n, nstep=NSTEP, warmup=4):
    head = list(range(min(warmup, n)))
    rest = n - len(head) - 1
    if rest <= 0: return sorted(set(head + [nstep - 1]))
    span = nstep - 1 - head[-1]
    body = [head[-1] + int(round((k + 1) * span / (rest + 1))) for k in range(rest)]
    return sorted(set(head + body + [nstep - 1]))

pipe = CogVideoXPipeline.from_pretrained(M, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True); pipe.vae.enable_tiling()
tf = pipe.transformer; orig = tf.forward
prompts = open("/root/viewdit/resources/prompts/prompt_video.txt").read().splitlines()

def run(pi, nfull, outdir):
    full = sched(nfull) if nfull < NSTEP else list(range(NSTEP))
    fs = set(full); st = {"i": -1, "last": None, "nfe": 0}
    def fwd(*args, **kw):
        st["i"] += 1; i = st["i"]
        if i in fs or st["last"] is None:
            out = orig(*args, **kw)
            o = out[0] if isinstance(out, (tuple, list)) else out
            st["last"] = o.detach(); st["nfe"] += 1
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
    print("[done] p=%d n=%d nfe=%d" % (pi, nfull, st["nfe"]), flush=True)

for n in (30,):
    for pi in range(8, 20):
        run(pi, n, "results/cogunif/full")
print("UNIF_DONE", flush=True)
