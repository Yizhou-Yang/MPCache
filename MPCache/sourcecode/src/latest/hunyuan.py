"""HunyuanVideo adapter (13B). Not yet exercised end to end: the released RA harness
has no Hunyuan entry point, so this backend is the piece that makes the third
backbone runnable at all. Same three members as every other backend."""

from __future__ import annotations
from .base import Backend

NAME = "hunyuan"
DEFAULTS = {"height": 720, "width": 1280, "guidance": 6.0, "negative": "", "fps": 24}


def load(model_path: str, dtype: str = "bfloat16"):
    import torch
    from diffusers import HunyuanVideoPipeline
    pipe = HunyuanVideoPipeline.from_pretrained(model_path, torch_dtype=getattr(torch, dtype))
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate(pipe, prompt: str, frames: int, steps: int, seed: int, **kw):
    import torch
    g = torch.Generator("cuda").manual_seed(seed)
    out = pipe(prompt=prompt, height=kw.get("height", DEFAULTS["height"]),
               width=kw.get("width", DEFAULTS["width"]), num_frames=frames,
               num_inference_steps=steps, guidance_scale=kw.get("guidance", DEFAULTS["guidance"]),
               generator=g)
    return out.frames[0]


backend = Backend(NAME, load, "transformer", generate, DEFAULTS)
