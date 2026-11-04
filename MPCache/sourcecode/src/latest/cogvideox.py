"""CogVideoX-2B adapter: about twenty lines, all of it about *this* model."""

from __future__ import annotations
from .base import Backend

NAME = "cogvideox"
DEFAULTS = {"height": 480, "width": 720, "guidance": 6.0, "negative": "animation", "fps": 8}


def load(model_path: str, dtype: str = "float16"):
    import torch
    from diffusers import CogVideoXPipeline
    pipe = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=getattr(torch, dtype)).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()
    return pipe


def generate(pipe, prompt: str, frames: int, steps: int, seed: int, **kw):
    import torch
    g = torch.Generator("cuda").manual_seed(seed)
    out = pipe(prompt=prompt, negative_prompt=kw.get("negative", DEFAULTS["negative"]),
               num_frames=frames, height=kw.get("height", DEFAULTS["height"]),
               width=kw.get("width", DEFAULTS["width"]), num_inference_steps=steps,
               guidance_scale=kw.get("guidance", DEFAULTS["guidance"]), generator=g)
    return out.frames[0]


backend = Backend(NAME, load, "transformer", generate, DEFAULTS)
