"""Bilevel schedule + causal chord-shrinkage runtime controller.

Drives `wrap_forward` across CogVideoX, Wan, HunyuanVideo, or unit-test models via
`decide() / record(out) / reuse() / verify()` (plus optional callable shorthand).
When `alpha == 0.0` and `policy in ("reuse", "damped")` it performs exact zero-copy
reuse of `last_exact`; setting `alpha = 0.5, policy = "damped"` activates the
Theorem 3 causal midpoint chord-shrinkage operator `v_s + alpha * (v_s - v_{s'})`.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .shrinkage import ChordShrinkagePredictor


@dataclass
class ScheduleController:
    schedule: Iterable[int]
    alpha: float = 0.0
    policy: str = "damped"
    regime_head: int = 0
    last_exact: Any = None
    step: int = -1
    calls: int = 0
    decisions: list[tuple[int, str]] = field(default_factory=list)
    strict: bool = True
    predictor: ChordShrinkagePredictor = field(init=False)

    def __post_init__(self) -> None:
        self.schedule = set(int(s) for s in self.schedule)
        if not self.schedule:
            raise ValueError("empty schedule")
        self.predictor = ChordShrinkagePredictor(
            policy=self.policy, alpha=self.alpha, regime_head=self.regime_head
        )

    def reset(self) -> None:
        self.last_exact = None
        self.step = -1
        self.calls = 0
        self.decisions.clear()
        self.predictor.reset()

    def decide(self) -> bool:
        """Advance one step and return True if the caller must evaluate."""
        self.step += 1
        evaluate = self.step in self.schedule or self.last_exact is None
        self.decisions.append((self.step, "eval" if evaluate else "reuse"))
        return evaluate

    def record(self, out: Any) -> Any:
        """Store the exact output of the step just evaluated."""
        self.last_exact = out.detach() if hasattr(out, "detach") else out
        self.calls += 1
        self.predictor.observe_exact(self.step, self.last_exact)
        return out

    def reuse(self) -> Any:
        """Return either plain `last_exact` or the phase-gated chord-shrinkage estimate."""
        if self.last_exact is None:
            raise RuntimeError("reuse before any exact step")
        if self.policy == "reuse" or (
            self.policy in ("damped", "midpoint", "phase_gated") and self.alpha == 0.0
        ):
            return self.last_exact
        return self.predictor.predict(self.step)

    def __call__(self, fn: Callable[[], Any]) -> Any:
        """Convenience callable wrapper equivalent to `decide` + `record`/`reuse`."""
        if self.decide():
            return self.record(fn())
        return self.reuse()

    def verify(self) -> None:
        """Check invariants a run must satisfy; call after generation."""
        if not self.strict:
            return
        expected = len(self.schedule) + (0 if 0 in self.schedule else 1)
        if self.calls != expected:
            raise AssertionError(f"calls {self.calls} != expected {expected}")
        if self.step + 1 < max(self.schedule):
            raise AssertionError("schedule contains steps the run never reached")
