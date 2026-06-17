"""Text-table report generation from ModuleEvent lists."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from tabulate import tabulate

from glasstrace.hooks import ModuleEvent, Phase


def _aggregate(events: list[ModuleEvent]) -> list[dict]:
    """Aggregate events by module path: sum times, count calls."""
    agg: dict[str, dict] = defaultdict(
        lambda: {"calls": 0, "total_ms": 0.0, "module_type": "", "device": ""}
    )
    for e in events:
        a = agg[e.module_path]
        a["calls"] += 1
        a["total_ms"] += e.duration_ms
        a["module_type"] = e.module_type
        a["device"] = e.device
    return [
        {"path": path, **vals}
        for path, vals in sorted(
            agg.items(), key=lambda x: x[1]["total_ms"], reverse=True
        )
    ]


def _section_table(rows: list[dict], total_ms: float, extra_col: str | None = None) -> str:
    """Format a list of aggregated module rows as a text table."""
    if not rows:
        return "  (no events)\n"

    table_rows = []
    for r in rows:
        row = {
            "Module": r["path"],
            "Type": r["module_type"],
            "Calls": r["calls"],
            "Total ms": f"{r['total_ms']:.2f}",
            "Per-call ms": f"{r['total_ms'] / r['calls']:.2f}",
            "% of phase": f"{r['total_ms'] / total_ms * 100:.1f}" if total_ms > 0 else "—",
        }
        table_rows.append(row)

    return tabulate(table_rows, headers="keys", tablefmt="simple") + "\n"


def format_report(
    events: Iterable[ModuleEvent],
    memory_samples: list[dict] | None = None,
    top_n: int = 20,
) -> str:
    """Produce a two-section report: prefill and decode."""
    events = list(events)
    if not events:
        return (
            "glasstrace: no events recorded.\n"
            "(Was the model actually run inside the profile() block?)"
        )

    device = events[0].device

    prefill = [e for e in events if e.phase == Phase.PREFILL]
    decode = [e for e in events if e.phase == Phase.DECODE]
    unknown = [e for e in events if e.phase == Phase.UNKNOWN]

    prefill_ms = sum(e.duration_ms for e in prefill)
    decode_ms = sum(e.duration_ms for e in decode)
    total_ms = sum(e.duration_ms for e in events)

    # Decode passes = number of unique decode events for one module
    # (all modules fire once per decode token)
    decode_passes = decode[0].module_path and len(
        [e for e in decode if e.module_path == decode[0].module_path]
    ) if decode else 0
    per_token_ms = decode_ms / decode_passes if decode_passes > 0 else 0.0

    # Memory summary
    mem_summary = ""
    if memory_samples:
        decode_samples = [s for s in memory_samples if s["phase"] == "decode"]
        if decode_samples:
            min_mem = min(s["memory_bytes"] for s in decode_samples)
            max_mem = max(s["memory_bytes"] for s in decode_samples)
            kv_growth_mb = (max_mem - min_mem) / (1024 ** 2)
            mem_summary = f"  kv-cache growth during decode: {kv_growth_mb:.1f} MB\n"

    header = (
        f"\nglasstrace report\n"
        f"  modules profiled: {len({e.module_path for e in events})}\n"
        f"  total events: {len(events)}\n"
        f"  total measured time: {total_ms:.2f} ms\n"
        f"  device: {device}\n"
        + mem_summary
    )

    # Prefill section
    prefill_header = (
        f"\n── prefill (1 pass, {prefill_ms:.1f} ms total) "
        + "─" * 40 + "\n"
    )
    prefill_rows = _aggregate(prefill)[:top_n]
    prefill_table = _section_table(prefill_rows, prefill_ms)

    # Decode section
    decode_header = (
        f"\n── decode ({decode_passes} passes, {decode_ms:.1f} ms total"
        + (f", {per_token_ms:.1f} ms/token avg" if per_token_ms > 0 else "")
        + ") " + "─" * 20 + "\n"
    )
    decode_rows = _aggregate(decode)[:top_n]
    decode_table = _section_table(decode_rows, decode_ms)

    # Unknown section (should be empty for standard transformer runs)
    unknown_section = ""
    if unknown:
        unknown_ms = sum(e.duration_ms for e in unknown)
        unknown_section = (
            f"\n── unclassified ({len(unknown)} events, {unknown_ms:.1f} ms) ──\n"
        )

    return (
        header
        + prefill_header
        + prefill_table
        + decode_header
        + decode_table
        + unknown_section
    )
