"""Turn power samples in a CSV into joules over a wall-clock window."""

import csv
import datetime
import itertools
import pathlib


def window_energy(
    path: pathlib.Path,
    start: datetime.datetime,
    stop: datetime.datetime,
    source: str,
    baseline_w: float = 0.0,
) -> dict[str, float | int]:
    """Integrate one CSV's power over a wall-clock window.

    The GPU file carries an energy counter that the hardware integrates, so
    joules come from it; the system file has no counter, so apply the
    trapezoid rule to its power samples instead.

    Args:
        path: A CSV from the samplers.
        start: First timestamp to include.
        stop: Last timestamp to include.
        source: `"gpu"` or `"system"`, and the file is checked against it.
        baseline_w: Idle draw to subtract for the incremental figures.

    Returns:
        Total and incremental joules, mean watts, and the window measured.
    """
    # Keep the rows that fall inside the window.
    is_gpu = source == "gpu"
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            stamp = datetime.datetime.fromisoformat(r["timestamp"])
            if start <= stamp <= stop:
                counter = int(r["energy_mj"]) if is_gpu else None
                rows.append((stamp, float(r["power_w"]), counter))

    if len(rows) < 2:
        raise ValueError(f"{path}: {len(rows)} rows in window, need 2")

    # Integrate the power column, and read how far the counter advanced.
    (t_first, _, c_first), (t_last, _, c_last) = rows[0], rows[-1]
    seconds = (t_last - t_first).total_seconds()
    integrated = sum(
        (p0 + p1) / 2 * (t1 - t0).total_seconds()
        for (t0, p0, _), (t1, p1, _) in itertools.pairwise(rows)
    )
    joules = (c_last - c_first) / 1000 if is_gpu else integrated

    # Report the totals and the rates they imply.
    return {
        "joules": joules,
        "joules_incremental": joules - baseline_w * seconds,
        "mean_w": joules / seconds,
        "n_samples": len(rows),
        "coverage_s": seconds,
    }
