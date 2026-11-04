import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mpcache.controller import ScheduleController
from mpcache.instrument import wrap_forward


class Toy:
    def __init__(self): self.calls = 0
    def forward(self, x):
        self.calls += 1
        return ("v%d" % self.calls,)


def run(schedule, steps):
    t, c = Toy(), ScheduleController(schedule)
    restore = wrap_forward(t, c)
    outs = [t.forward(0) for _ in range(steps)]
    restore()
    return t, c, outs


def test_budget_is_exact_and_reuse_is_cached():
    t, c, outs = run([0, 2, 4], 5)
    assert t.calls == 3 and c.calls == 3
    assert outs[1][0] == outs[0][0] and outs[3][0] == outs[2][0]
    c.verify()


def test_first_step_always_evaluates_even_if_unscheduled():
    t, c, outs = run([3], 5)
    assert t.calls == 2          # step 0 forced, then step 3
    c.verify()


def test_restore_leaves_no_trace():
    t, c, _ = run([0, 1], 2)
    before = t.calls
    t.forward(0)
    assert t.calls == before + 1  # unpatched again


def test_budget_mismatch_is_caught():
    t, c, _ = run([0, 1, 2], 2)   # schedule has more steps than the run
    try:
        c.verify()
    except AssertionError:
        return
    raise AssertionError("verify should have failed")
