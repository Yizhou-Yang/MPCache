"""The step-decision state machine, independent of any benchmark.

A controller knows the schedule (which step indices to evaluate), counts the calls
it has seen, and answers one question per step: evaluate, or reuse the last exact
output. It owns no tensors itself and imports no framework, so the same object
drives CogVideoX, Wan, HunyuanVideo or a toy model in a test.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ScheduleController:
    schedule: set[int]
    last_exact: object | None = None
    step: int = -1
    calls: int = 0
    decisions: list[tuple[int, str]] = field(default_factory=list)
    strict: bool = True          # assert that the schedule is consumed exactly

    def __post_init__(self):
        self.schedule = set(int(s) for s in self.schedule)
        if not self.schedule:
            raise ValueError("empty schedule")

    def decide(self) -> bool:
        """Advance one step and return True if the caller must evaluate."""
        self.step += 1
        evaluate = self.step in self.schedule or self.last_exact is None
        self.decisions.append((self.step, "eval" if evaluate else "reuse"))
        return evaluate

    def record(self, out):
        """Store the exact output of the step just evaluated."""
        self.last_exact = out.detach() if hasattr(out, "detach") else out
        self.calls += 1
        return out

    def reuse(self):
        if self.last_exact is None:
            raise RuntimeError("reuse before any exact step")
        return self.last_exact

    def verify(self) -> None:
        """Check the invariants a run must satisfy; call after generation."""
        if not self.strict:
            return
        # the first step has no cached output to reuse, so it is evaluated even when
        # the schedule does not list it; every schedule we ship lists step 0 anyway.
        expected = len(self.schedule) + (0 if 0 in self.schedule else 1)
        if self.calls != expected:
            raise AssertionError(f"calls {self.calls} != expected {expected}")
        if self.step + 1 < max(self.schedule):
            raise AssertionError("schedule contains steps the run never reached")
