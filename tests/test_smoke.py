"""Smoke tests — verify the package imports and the profiler runs end-to-end."""

import torch
import torch.nn as nn

import glasstrace
from glasstrace.hooks import Phase


def test_import():
    assert glasstrace is not None
    assert glasstrace.profile is not None


def test_version():
    assert isinstance(glasstrace.__version__, str)
    assert len(glasstrace.__version__) > 0


def test_profile_tiny_model():
    """Profiler captures events when a tiny model is run inside the context."""

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.norm = nn.LayerNorm(16)
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            return self.fc2(self.norm(self.fc1(x)))

    model = Tiny()
    x = torch.randn(2, 8)

    with glasstrace.profile(model) as p:
        _ = model(x)

    assert len(p) >= 3
    module_paths = {e.module_path for e in p.events}
    assert "fc1" in module_paths
    assert "norm" in module_paths
    assert "fc2" in module_paths
    for e in p.events:
        assert e.duration_ms >= 0


def test_report_format():
    """Report renders something non-empty for a small profile."""
    model = nn.Sequential(nn.Linear(4, 8), nn.LayerNorm(8), nn.Linear(8, 2))
    x = torch.randn(1, 4)

    with glasstrace.profile(model) as p:
        _ = model(x)

    report = p.report()
    assert isinstance(report, str)
    assert "glasstrace report" in report
    assert "Module" in report


def test_phase_detection():
    """Events are tagged with correct phase based on sequence dimension."""
    from glasstrace.hooks import ModuleTracer

    tracer = ModuleTracer()

    # seq_len > 1 → prefill
    assert tracer._detect_phase((2, 10, 64)) == Phase.PREFILL

    # seq_len == 1 → decode
    assert tracer._detect_phase((2, 1, 64)) == Phase.DECODE

    # No shape → unknown
    assert tracer._detect_phase(None) == Phase.UNKNOWN


def test_prefill_decode_split():
    """Profile of a sequence model separates prefill from decode events."""

    class SeqModel(nn.Module):
        """Simulates a tiny sequence model with variable-length inputs."""
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, x):
            return self.proj(x)

    model = SeqModel()

    with glasstrace.profile(model) as p:
        # Simulate prefill: batch=1, seq=5
        _ = model(torch.randn(1, 5, 8))
        # Simulate 3 decode steps: batch=1, seq=1 each
        for _ in range(3):
            _ = model(torch.randn(1, 1, 8))

    prefill_events = [e for e in p.events if e.phase == Phase.PREFILL]
    decode_events = [e for e in p.events if e.phase == Phase.DECODE]

    assert len(prefill_events) >= 1, "Expected at least one prefill event"
    assert len(decode_events) >= 3, "Expected at least three decode events"
