"""Compare closed-loop control with the open-loop optimum at a matched budget.

Same horizon program, same cost model, same profile; the only difference is whether
the drift rate is updated from what the run observes. CogVideoX-2B, single GPU.
"""
import argparse, json, os, sys
import numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from mpcache.control import Controller
from mpcache.metrics import compare

FIXED = set(list(range(16)) + [18, 22, 28, 35, 42, 49])
PROFILE_COG = [0.0,0.569,0.452,0.313,0.192,0.118,0.094,0.063,0.061,0.054,0.053,0.047,0.045,
               0.043,0.043,0.042,0.041,0.040,0.041,0.041,0.041,0.041,0.041,0.041,0.041,0.042,
               0.042,0.042,0.043,0.050]

def run_one(pipe, prompt, frames, steps, seed, budget, mode, rate0):
    tf = pipe.transformer; orig = tf.forward
    prof = np.interp(np.linspace(0, len(PROFILE_COG) - 1, steps),
                     np.arange(len(PROFILE_COG)), PROFILE_COG).tolist()
    ctrl = Controller(profile=prof, budget=budget, T=steps, rate=rate0, closed_loop=(mode == "mpc"))
    state = {"i": -1, "last": None, "calls": 0}
    def fwd(*a, **kw):
        state["i"] += 1; i = state["i"]
        left = budget - state["calls"]
        if mode == "dp":
            do = i in FIXED or state["last"] is None
        else:
            do = ctrl.decide(i, left) if state["last"] is not None else True
        if do:
            out = orig(*a, **kw)
            o = out[0] if isinstance(out, (tuple, list)) else out
            ctrl.observe(i, o); state["last"] = o; state["calls"] += 1
            return out
        return (state["last"],)
    tf.forward = fwd
    g = torch.Generator("cuda").manual_seed(seed)
    try:
        with torch.no_grad():
            out = pipe(prompt=prompt, negative_prompt="animation", num_frames=frames,
                       height=480, width=720, num_inference_steps=steps, guidance_scale=6.0, generator=g)
    finally:
        tf.forward = orig
    return out.frames[0], state["calls"], ctrl.rate

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--prompts", required=True)
    ap.add_argument("--frames", type=int, default=17); ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--budget", type=int, default=22); ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--rate0", type=float, default=0.02)
    ap.add_argument("--out", required=True); ap.add_argument("--ledger", default=None)
    a = ap.parse_args()
    from diffusers import CogVideoXPipeline
    import imageio.v2 as iio
    pipe = CogVideoXPipeline.from_pretrained(a.model, torch_dtype=torch.float16).to("cuda")
    pipe.set_progress_bar_config(disable=True); pipe.vae.enable_tiling()
    prompts = [l.strip() for l in open(a.prompts) if l.strip()][:a.limit]
    for mode in ("dp", "mpc"):
        d = os.path.join(a.out, mode); os.makedirs(d, exist_ok=True)
        rates = []
        for i, p in enumerate(prompts):
            frames, calls, rate = run_one(pipe, p, a.frames, a.steps, i, a.budget, mode, a.rate0)
            rates.append(rate)
            arr = np.stack([np.asarray(f).astype("uint8") for f in frames])
            iio.mimsave(os.path.join(d, "video_%06d.mp4" % i), arr, fps=8, codec="libx264", macro_block_size=8)
            print("[%s] %d calls=%d rate=%.4f" % (mode, i, calls, rate), flush=True)
        print("[%s] mean rate %.4f" % (mode, float(np.mean(rates))), flush=True)
    for mode in ("dp", "mpc"):
        r = compare(os.path.join(a.out, "dp"), os.path.join(a.out, mode), 10, "vgg")
        print(mode, r, flush=True)
        if a.ledger:
            from mpcache.ledger import record
            record(a.ledger, "cog17_%s" % mode, **r)

if __name__ == "__main__":
    main()
