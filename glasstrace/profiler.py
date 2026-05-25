"""The user-facing profile() context manager."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch.nn as nn

from glasstrace.hooks import ModuleEvent, ModuleTracer
from glasstrace.report import format_report


@dataclass
class ProfileResult:
    """Holds events collected during a profile() block. Exposes report helpers."""

    events: list[ModuleEvent] = field(default_factory=list)

    def report(self, top_n: int = 20) -> str:
        """Return a formatted text-table report."""
        return format_report(self.events, top_n=top_n)

    def __len__(self) -> int:
        return len(self.events)


@contextmanager
def profile(model: nn.Module):
    """Profile a model's forward passes within a with-block.

    Example:
        with glasstrace.profile(model) as p:
            model.generate(**inputs, max_new_tokens=50)
        print(p.report())"""

    tracer = ModuleTracer()
    tracer.attach(model)
    result = ProfileResult(events=tracer.events)
    try:
        yield result
    finally:
        tracer.detach()
