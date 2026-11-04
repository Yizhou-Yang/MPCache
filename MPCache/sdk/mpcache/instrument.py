"""Attach a controller to any `forward` method, without touching the model.

The wrapper is benchmark-agnostic: it counts calls, returns the cached output on a
skipped step, and passes everything else through. It only assumes that the value
returned to the pipeline is either a tensor or a one-element tuple of tensors,
which is the case for every transformer block in the pipelines we use.
"""

from __future__ import annotations


def _first(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _rewrap(template, value):
    return (value,) if isinstance(template, (tuple, list)) else value


def wrap_forward(module, controller, attr: str = "forward"):
    """Patch `module.<attr>` so that skipped steps return the cached output.

    Returns a function that restores the original method. The module is untouched
    otherwise: no hooks, no buffers, no state_dict changes.
    """
    original = getattr(module, attr)
    last_shape = {"template": None}

    def wrapped(*args, **kwargs):
        if controller.decide():
            out = original(*args, **kwargs)
            controller.record(_first(out))
            last_shape["template"] = out
            return out
        cached = controller.reuse()
        return _rewrap(last_shape["template"], cached)

    setattr(module, attr, wrapped)

    def restore():
        setattr(module, attr, original)

    return restore
