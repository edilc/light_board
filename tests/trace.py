"""Helpers for printing recorded effect traces in a way that makes the curve
shape obvious at a glance. Used by tests and ad-hoc debugging."""
from __future__ import annotations

Color = tuple[int, int, int]
Snapshot = tuple[Color, ...]
Event = tuple[float, Snapshot]


def summarize(name: str, events: list[Event]) -> str:
    """One-line stats. For uniform traces, reports channel 0 only; for
    non-uniform traces, notes that the channels differ."""
    if not events:
        return f"{name}: no events"
    total = len(events)
    span = events[-1][0] - events[0][0]

    uniform = all(len(set(snap)) == 1 for _, snap in events)

    # Stats from channel 0 (most useful for uniform effects).
    ch0 = [(t, *snap[0]) for t, snap in events]
    max_r = max(e[1] for e in ch0)
    max_g = max(e[2] for e in ch0)
    max_b = max(e[3] for e in ch0)
    first_nonzero = next(
        (i for i, (_, r, g, b) in enumerate(ch0) if (r, g, b) != (0, 0, 0)),
        None,
    )
    first_visible = next(
        (i for i, (_, r, g, b) in enumerate(ch0) if max(r, g, b) >= 10),
        None,
    )
    fnz_t = ch0[first_nonzero][0] - ch0[0][0] if first_nonzero is not None else None
    fv_t = ch0[first_visible][0] - ch0[0][0] if first_visible is not None else None
    suffix = "" if uniform else f" | non-uniform across {len(events[0][1])} channels"
    return (
        f"{name}: {total} frames over {span:.2f}s | "
        f"ch0 max=({max_r},{max_g},{max_b}) | "
        f"first non-zero @ idx {first_nonzero} ({fnz_t}s) | "
        f"first ≥10 @ idx {first_visible} ({fv_t}s) | "
        f"last_ch0={ch0[-1][1:]}{suffix}"
    )


def sparse_table(events: list[Event], every: int) -> str:
    """Print every Nth event plus the last one. Auto-collapses to one column
    when all snapshots are uniform across channels; otherwise shows a column
    per channel."""
    if not events:
        return "(no events)"

    uniform = all(len(set(snap)) == 1 for _, snap in events)
    n_channels = len(events[0][1])
    rows: list[str] = []

    if uniform:
        rows.append(f"{'idx':>5}  {'t (s)':>7}  {'R':>4} {'G':>4} {'B':>4}")
        rows.append("-" * 30)
        t0 = events[0][0]
        for i, (t, snap) in enumerate(events):
            if i % every == 0 or i == len(events) - 1:
                r, g, b = snap[0]
                rows.append(f"{i:>5}  {t - t0:>7.2f}  {r:>4} {g:>4} {b:>4}")
    else:
        header = f"{'idx':>5}  {'t (s)':>7}"
        for ch in range(n_channels):
            header += f"  ch{ch}: {'R':>4} {'G':>4} {'B':>4}"
        rows.append(header)
        rows.append("-" * len(header))
        t0 = events[0][0]
        for i, (t, snap) in enumerate(events):
            if i % every == 0 or i == len(events) - 1:
                line = f"{i:>5}  {t - t0:>7.2f}"
                for ch in range(n_channels):
                    r, g, b = snap[ch]
                    line += f"      {r:>4} {g:>4} {b:>4}"
                rows.append(line)
    return "\n".join(rows)
