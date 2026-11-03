"""How the schedule is injected into a released harness.

The paper's numbers come from the RA-CFGCache release.  Two edits are enough:
both put the same block at the top of the branch that decides whether a step is
evaluated, and both read the schedule from an environment variable, so nothing
else in the pipeline changes.
"""

COG_HOOK = '''\
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
'''

WAN_HOOK = '''\
import os as _os
_sched = _os.environ.get("MPC_SCHEDW", "")
if _sched:
    _ss = set(int(x) for x in _sched.split(",") if x)
    _st = int(current.get("step", 0))
    if _st in _ss:
        current["type"] = "full"
        cache_dic["full_counter"] = cache_dic.get("full_counter", 0) + 1
        return
    current["type"] = "CFGCacheSkip"
    return
'''


def inject(path: str, hook: str, anchor: str) -> None:
    """Insert `hook` right after the first line that starts with `anchor`."""
    with open(path) as f:
        src = f.read()
    if hook.strip().splitlines()[0] in src:
        return
    idx = src.index(anchor) + len(anchor)
    with open(path, "w") as f:
        f.write(src[:idx] + hook + src[idx:])
