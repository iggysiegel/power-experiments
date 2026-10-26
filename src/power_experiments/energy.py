"""Turn power samples in a CSV into joules over a wall-clock window."""

import datetime
import pathlib

import pandas as pd


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
    df = pd.read_csv(path, parse_dates=["timestamp"])
    df = df[df.timestamp.between(start, stop)]

    if len(df) < 2:
        raise ValueError(f"{path}: {len(df)} rows in window, need 2")

    # Read how far the counter advanced, or integrate the power column.
    elapsed = (df.timestamp - df.timestamp.iloc[0]).dt.total_seconds()
    seconds = elapsed.iloc[-1]
    if source == "gpu":
        joules = (df.energy_mj.iloc[-1] - df.energy_mj.iloc[0]) / 1000
    else:
        # Trapezoid rule: each interval at the mean of its two ends.
        joules = (df.power_w.rolling(2).mean() * elapsed.diff()).sum()

    # Report the totals and the rates they imply.
    return {
        "joules": joules,
        "joules_incremental": joules - baseline_w * seconds,
        "mean_w": joules / seconds,
        "n_samples": len(df),
        "coverage_s": seconds,
    }
