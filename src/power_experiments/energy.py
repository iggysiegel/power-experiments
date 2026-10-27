"""Turn power samples in a CSV into joules over a wall-clock window."""

import datetime
import pathlib

import numpy as np
import pandas as pd


def window_energy(
    path: pathlib.Path,
    start: datetime.datetime,
    stop: datetime.datetime,
    source: str,
    baseline_w: float = 0.0,
) -> dict[str, float]:
    """Integrate one CSV's power over exactly a wall-clock window.

    The GPU file carries an energy counter that the hardware integrates, so
    joules come from it; the system file has no counter, so apply the
    trapezoid rule to its power samples instead. Both are interpolated at the
    window's edges, and carried on at the nearest sample's power where the
    trace stops short of an edge.

    Args:
        path: A CSV from the samplers.
        start: Start of the window.
        stop: End of the window.
        source: `"gpu"` or `"system"`, saying which sampler wrote the file.
        baseline_w: Idle draw to subtract for the incremental figures.

    Returns:
        Total and incremental joules, and mean watts.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    seconds = (stop - start).total_seconds()
    elapsed = (df.timestamp - start).dt.total_seconds().to_numpy()
    power_w = df.power_w.to_numpy()

    if source == "gpu":
        # Read the counter at each edge, interpolating between samples, and
        # past either end of the trace carry it on at the end's power.
        counter_j = df.energy_mj.to_numpy() / 1000
        edges = np.array([0.0, seconds])
        before = np.minimum(edges - elapsed[0], 0)
        after = np.maximum(edges - elapsed[-1], 0)
        at_edges = (
            np.interp(edges, elapsed, counter_j)
            + power_w[0] * before
            + power_w[-1] * after
        )
        joules = at_edges[1] - at_edges[0]
    else:
        # Trapezoid rule over the samples inside plus one point at each edge,
        # whose power np.interp interpolates, or holds past the trace's ends.
        inside = (elapsed > 0) & (elapsed < seconds)
        t = np.concatenate([[0.0], elapsed[inside], [seconds]])
        joules = np.trapezoid(np.interp(t, elapsed, power_w), t)

    # Report the totals and the rate they imply.
    return {
        "joules": joules,
        "joules_incremental": joules - baseline_w * seconds,
        "mean_w": joules / seconds,
    }
