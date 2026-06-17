"""The user-facing profile() context manager."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Callable

import torch.nn as nn

from glasstrace.hooks import ModuleEvent, ModuleTracer
from glasstrace.report import format_report


@dataclass
class ProfileResult:
    """Holds events and memory samples from a profile() block."""

    events: list[ModuleEvent] = field(default_factory=list)
    memory_samples: list[dict] = field(default_factory=list)

    def report(self, top_n: int = 20) -> str:
        """Return a formatted two-section text report."""
        return format_report(self.events, self.memory_samples, top_n=top_n)

    def __len__(self) -> int:
        return len(self.events)


@contextmanager
def profile(model: nn.Module, warmup: Callable[[], None] | None = None):
    """Profile a model's forward passes within a with-block.

    Args:
        model: the model to instrument.
        warmup: optional zero-arg callable run once before profiling starts,
            with its events discarded. Strongly recommended on CUDA to avoid
            cold-start timing artifacts.

    Example:
        def warmup():
            model.generate(**inputs, max_new_tokens=5)

        with glasstrace.profile(model, warmup=warmup) as p:
            model.generate(**inputs, max_new_tokens=50)
        print(p.report())
    """
    if warmup is not None:
        import torch
        with torch.no_grad():
            warmup()
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    tracer = ModuleTracer()
    tracer.attach(model)
    result = ProfileResult(events=tracer.events, memory_samples=tracer.memory_samples)
    try:
        yield result
    finally:
        tracer.detach()
