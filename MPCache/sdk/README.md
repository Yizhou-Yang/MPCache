# sdk/ — the benchmark-agnostic layer

Everything that is the same for every video diffusion model lives here. A benchmark
contributes **one file of about twenty lines** under `mpcache/backends/`; nothing else
is model-specific.

```
mpcache/
  schedule.py     the allocation solver: greedy on a measured profile, and the geometric family
  controller.py   the step state machine: counts calls, decides evaluate or reuse, holds the cache
  instrument.py   wraps any forward method; no hooks, no state_dict changes
  profile.py      the single-skip measurement protocol, parameterised by a `run` callable
  metrics.py      LPIPS(VGG) / SSIM / PSNR over matched videos
  ledger.py       append results to JSONL the moment they exist
  runner.py       the one runner every benchmark uses
  backends/       the only place a benchmark is named
```

## Why this split

The parts that vary between CogVideoX, Wan and HunyuanVideo are: the pipeline class,
the attribute that holds the transformer, and the call signature of generation. That
is it. The schedule, the decision logic, the caching, the measurement and the metrics
are identical, so they are written once and tested once.

## Adding a benchmark

```python
# mpcache/backends/myvideo.py
from .base import Backend
NAME = "myvideo"; DEFAULTS = {"height": 480, "width": 720, "guidance": 7.0, "negative": "", "fps": 8}
def load(path, dtype="bfloat16"): ...
def generate(pipe, prompt, frames, steps, seed, **kw): ...
backend = Backend(NAME, load, "transformer", generate, DEFAULTS)
```

Register it in `backends/__init__.py` and it is immediately runnable through
`runner.py`, with the controller, the instrumentation, the ledger and the metrics
already wired.

## The runner

```bash
python -m mpcache.runner --backend cogvideox --model /weights/CogVideoX-2b \
  --prompts prompts.txt --frames 49 --steps 50 \
  --schedule 0,1,2,3,4,5,6,7,8,9,10,11,13,15,18,22,27,33,40,48 \
  --out out/ours --ledger metrics.jsonl
```

The same command with `--backend wan` or `--backend hunyuan` runs the other backbones.
