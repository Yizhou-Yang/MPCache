"""A backend is the only place a benchmark is named.

Three members: how to load the pipeline, which attribute is the transformer, and
how to call generation. Everything else - the schedule, the controller, the
instrumentation, the metrics - lives in the SDK and is shared.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable


@dataclass
class Backend:
    name: str
    load: Callable            # (model_path) -> pipeline
    transformer: str          # attribute of the pipeline that we instrument
    generate: Callable        # (pipe, prompt, frames, steps, seed, **kw) -> frames
    defaults: dict            # height, width, guidance, negative prompt
