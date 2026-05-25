"""

Smoke tests — verify the package imports and the profiler runs end-to-end.

"""

import torch
import torch.nn as nn

import glasstrace


def test_import():
    """Package imports without errors."""
    assert glasstrace is not None
    assert glasstrace.profile is not None


def test_version():
    """Package exposes a version string."""
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

    # We expect at least 3 events: fc1 (Linear), norm (LayerNorm), fc2 (Linear)
    assert len(p) >= 3, f"Expected at least 3 events, got {len(p)}"

    module_paths = {e.module_path for e in p.events}
    assert "fc1" in module_paths
    assert "norm" in module_paths
    assert "fc2" in module_paths

    # Every event should have positive duration
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
