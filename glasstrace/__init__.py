"""glasstrace — per-layer profiler for transformer inference."""

from glasstrace.profiler import ProfileResult, profile

__version__ = "0.3.0"

__all__ = ["__version__", "profile", "ProfileResult"]
