import os, torch, numpy as np, imageio.v2 as iio
from diffusers import CogVideoXPipeline
M = "/root/viewdit/weights/CogVideoX-2b"

def sched(n_full, nstep, warmup=4):
    head = list(range(min(warmup, n_full)))
    rest = n_full - len(head) - 1
    if rest <= 0: return sorted(set(head + [nstep - 1]))
    span = nstep - 1 - head[-1]
    body = [head[-1] + int(round((k + 1) * span / (rest + 1))) for k in range(rest)]
    return sorted(set(head + body + [nstep - 1]))

pipe = CogVideoXPipeline.from_pretrained(M, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True)
pipe.vae.enable_tiling()
tf = pipe.transformer
orig = tf.forward
prompts = open("/root/viewdit/resources/prompts/prompt_video.txt").read().splitlines()

def run(pi, nfull, mode, outdir, alpha=1.0):
    full = sched(nfull, 30) if nfull < 30 else list(range(30))
    st = {"i": -1, "outs": []}
    fs = set(full)
    def fwd(*args, **kw):
        st["i"] += 1; i = st["i"]
        if i in fs or not st["outs"]:
            out = orig(*args, **kw)
            o = out[0] if isinstance(out, (tuple, list)) else out
            st["outs"].append((i, o.detach()))
            if len(st["outs"]) > 2: st["outs"].pop(0)
            return out
        i1, o1 = st["outs"][-1]
        if mode == "reuse" or len(st["outs"]) < 2:
            return (o1,)
        i0, o0 = st["outs"][-2]
        r = (i - i1) / max(1, (i1 - i0))
        return (o1 + alpha * r * (o1 - o0),)
    tf.forward = fwd
    g = torch.Generator("cuda").manual_seed(pi)
    with torch.no_grad():
        out = pipe(prompt=prompts[pi], negative_prompt="animation", num_frames=17,
                   height=480, width=720, num_inference_steps=30, guidance_scale=6.0, generator=g)
    tf.forward = orig
    frames = np.stack([np.array(f).astype(np.uint8) for f in out.frames[0]])
    os.makedirs(outdir, exist_ok=True)
    iio.mimsave(os.path.join(outdir, "video_%06d.mp4" % pi), frames, fps=8, codec="libx264", macro_block_size=8)
    print("[done] p=%d nfe=%d mode=%s" % (pi, len(full), mode), flush=True)

for pi in range(8):
    for al in (0.25, 0.5):
        run(pi, 19, "ex1", "results/cog30/a%s" % str(al).replace(".", ""), alpha=al)
    print("[prompt]", pi, flush=True)
print("SWEEP_DONE", flush=True)
