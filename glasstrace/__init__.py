"""glasstrace — per-layer profiler for transformer inference."""

from glasstrace.html_report import save_html as save_html_report
from glasstrace.profiler import ProfileResult, profile

__version__ = "0.4.0"

__all__ = ["__version__", "profile", "ProfileResult", "save_html_report"]
