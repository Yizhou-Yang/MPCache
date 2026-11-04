# The exact injection used for the paper's numbers

The paper's numbers come from the released RA-CFGCache pipeline, not from the SDK runner.
Two edits were made, both reading the schedule from an environment variable, so nothing
else in their pipeline changes. This file records those edits; `sdk/mpcache/instrument.py`
is the general version of the same idea.

```python
\
import os as _os
_sched = _os.environ.get("MPC_SCHED", "")
if _sched:
    _ss = set(int(x) for x in _sched.split(",") if x)
    _st = int(current.get("step", 0))
    if _st in _ss:
        current["type"] = "full"
        cache_dic["full_counter"] = cache_dic.get("full_counter", 0) + 1
        return
    current["type"] = "TeaCacheSkip"
    return
```

Wan uses the same block with `MPC_SCHEDW` and `CFGCacheSkip`.
