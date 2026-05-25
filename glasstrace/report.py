"""Text-table report generation from ModuleEvent lists."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from tabulate import tabulate

from glasstrace.hooks import ModuleEvent


def format_report(events: Iterable[ModuleEvent], top_n: int = 20) -> str:
    """Produce a human-readable summary table sorted by total time.

    Aggregates events by module_path: a module called N times appears once
    with summed time and call count."""
    events = list(events)
    if not events:
        return (
            "glasstrace: no events recorded.\n"
            "(Was the model actually run inside the profile() block?)"
        )

    aggregated: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "total_ms": 0.0, "module_type": "", "device": ""}
    )

    for e in events:
        agg = aggregated[e.module_path]
        agg["calls"] += 1
        agg["total_ms"] += e.duration_ms
        agg["module_type"] = e.module_type
        agg["device"] = e.device

    total_time = sum(a["total_ms"] for a in aggregated.values())

    rows = [
        {
            "Module": path,
            "Type": agg["module_type"],
            "Calls": agg["calls"],
            "Total ms": f"{agg['total_ms']:.2f}",
            "Per-call ms": f"{agg['total_ms'] / agg['calls']:.2f}",
            "% of total": f"{(agg['total_ms'] / total_time * 100):.1f}",
        }
        for path, agg in aggregated.items()
    ]
    rows.sort(key=lambda r: float(r["Total ms"]), reverse=True)
    rows = rows[:top_n]

    header_summary = (
        f"\nglasstrace report\n"
        f"  modules profiled: {len(aggregated)}\n"
        f"  total events: {len(events)}\n"
        f"  total measured time: {total_time:.2f} ms\n"
        f"  device: {events[0].device}\n"
    )
    table = tabulate(rows, headers="keys", tablefmt="simple")
    return header_summary + "\n" + table + "\n"
