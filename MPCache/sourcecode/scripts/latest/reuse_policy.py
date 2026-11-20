"""What a skipped step returns, at a fixed schedule and a fixed number of calls.

Policies
  full      no skipping (the reference)
  latest    the last exact output (what every cache in the literature does)
  extrap    v_s + alpha * (v_s - v_{s-1}); alpha does NOT depend on the gap
  scaled    v_s + gap * (v_s - v_{s-1})   (the textbook first-order choice)

The schedule is the one the paper ships: the first sixteen steps, then
18, 22, 28, 35, 42, 49, which is twenty-two transformer calls out of fifty.

Sharding: --offset and --limit select a prompt range and keep the global seed, so
several processes can write disjoint shards of one experiment. Outputs are named by
the global prompt index, so a recursive evaluation matches shards by basename.
"""
import os, argparse, numpy as np, torch, warnings, imageio.v2 as iio
warnings.filterwarnings("ignore")

FIXED = list(range(16)) + [18, 22, 28, 35, 42, 49]

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--prompts", required=True)
ap.add_argument("--frames", type=int, default=49); ap.add_argument("--steps", type=int, default=50)
ap.add_argument("--policy", required=True, choices=["full", "latest", "extrap", "scaled"])
ap.add_argument("--alpha", type=float, default=0.5)
ap.add_argument("--offset", type=int, default=0); ap.add_argument("--limit", type=int, default=8)
ap.add_argument("--out", required=True)
a = ap.parse_args()

from diffusers import CogVideoXPipeline
pipe = CogVideoXPipeline.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda")
pipe.set_progress_bar_config(disable=True); pipe.vae.enable_tiling()
tf = pipe.transformer; orig = tf.forward
allp = [l.strip() for l in open(a.prompts) if l.strip()]
prompts = allp[a.offset:a.offset + a.limit]
os.makedirs(a.out, exist_ok=True)

for i, p in enumerate(prompts):
    gi = a.offset + i
    st = {"k": -1, "cur": None, "prev": None, "anchor": -1, "calls": 0}
    def fwd(*x, **kw):
        st["k"] += 1; k = st["k"]
        do = True if (a.policy == "full" or st["cur"] is None) else (k in FIXED)
        if do:
            out = orig(*x, **kw); o = out[0] if isinstance(out, (tuple, list)) else out
            st["prev"], st["cur"] = st["cur"], (o.detach() if hasattr(o, "detach") else o)
            st["anchor"] = k; st["calls"] += 1
            return out
        if a.policy == "latest" or st["prev"] is None:
            return (st["cur"],)
        d = st["cur"] - st["prev"]
        if a.policy == "scaled":
            return (st["cur"] + max(1, k - st["anchor"]) * d,)
        return (st["cur"] + a.alpha * d,)
    tf.forward = fwd
    g = torch.Generator("cuda").manual_seed(gi)
    try:
        with torch.no_grad():
            out = pipe(prompt=p, negative_prompt="animation", num_frames=a.frames, height=480, width=720,
                       num_inference_steps=a.steps, guidance_scale=6.0, generator=g)
    finally:
        tf.forward = orig
    iio.mimsave(os.path.join(a.out, "video_%06d.mp4" % gi),
                np.stack([np.asarray(f).astype("uint8") for f in out.frames[0]]),
                fps=8, codec="libx264", macro_block_size=8)
    print("[%s a=%.2f] %d calls=%d" % (a.policy, a.alpha, gi, st["calls"]), flush=True)
print("SHARD_DONE", flush=True)
