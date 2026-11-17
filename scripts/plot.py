"""Plot each workload's energy curve with its fit, power during the restart,
and power across each experiment, into a campaign's directory.

Usage:
    uv run scripts/plot.py <campaign directory>
"""

import argparse
import pathlib
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches, ticker
from numpy.polynomial import polynomial

SOURCES = {"gpu": "GPU", "system": "Whole server"}

# The bench flag each workload varies across its runs.
SWEPT = {
    "prefill": "random-input-len",
    "decode": "max-concurrency",
    "agentic": "random-prefix-len",
}


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "campaign", type=pathlib.Path, help="campaign directory"
    )
    return parser.parse_args()


def read(run_dir: pathlib.Path, filename: str) -> pd.DataFrame:
    """Read one of a run's timestamped CSVs.

    Args:
        run_dir: The run's directory.
        filename: Such as `phases.csv` or `gpu.csv`.

    Returns:
        The CSV, with `timestamp` parsed.
    """
    return pd.read_csv(run_dir / filename, parse_dates=["timestamp"])


def workload_points(table: pd.DataFrame, name: str, per: str) -> pd.DataFrame:
    """Select one workload's runs and divide their energy by a count.

    Args:
        table: The results table.
        name: The workload's run name, such as `prefill`.
        per: Column to divide joules above idle by.

    Returns:
        The workload rows, with `gpu` and `system` holding joules above idle
        per unit of `per`.
    """
    rows = table[table.run.str.startswith(f"{name}-")]
    rows = rows[rows.phase == "workload"].copy()
    for source in SOURCES:
        rows[source] = rows[f"{source}_j_above"] / rows[per]
    return rows


def plot_with_fits(
    ax: plt.Axes,
    points: pd.DataFrame,
    x: str,
    basis: Callable,
    degree: int,
    equation: str,
) -> None:
    """Plot each source's points and polynomial fit, and print the fits.

    Args:
        ax: The plot to draw on.
        points: Rows from `workload_points`.
        x: Column holding the swept setting, drawn on a log2 axis.
        basis: Function of x the energy is a polynomial in, such as `1 / x`.
        degree: Degree of the polynomial.
        equation: The fitted form, printed above the coefficients in
            the order its terms appear.
    """
    swept = np.sort(points[x].unique())
    smooth = np.geomspace(swept[0], swept[-1], 200)
    print(equation)
    for i, (source, label) in enumerate(SOURCES.items()):
        # Values span decades, so fit relative error.
        fit = polynomial.polyfit(
            basis(points[x]), points[source], degree, w=1 / points[source]
        )
        print(f"  {source:<6}", "  ".join(f"{v:.4g}" for v in fit))
        ax.plot(points[x], points[source], "o", color=f"C{i}", label=label)
        ax.plot(
            smooth,
            polynomial.polyval(basis(smooth), fit),
            color=f"C{i}",
            linewidth=1,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(swept, [str(int(v)) for v in swept])
    ax.minorticks_off()
    ax.legend()


def plot_prefill(campaign: pathlib.Path, table: pd.DataFrame) -> None:
    """Plot prefill energy per request against input length.

    Args:
        campaign: The campaign's directory, to write the figure into.
        table: The campaign's results table.
    """
    points = workload_points(table, "prefill", "requests")
    fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
    plot_with_fits(
        ax,
        points,
        SWEPT["prefill"],
        lambda v: v,
        2,
        "prefill J/request = a + b * input + c * input^2",
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter("{x:g}")
    ax.minorticks_off()
    ax.set_xlabel("Input length (tokens)")
    ax.set_ylabel("Energy above idle per request (J)")
    fig.savefig(campaign / "prefill.png", dpi=200)


def plot_decode(campaign: pathlib.Path, table: pd.DataFrame) -> None:
    """Plot decode energy per output token against concurrency.

    Args:
        campaign: The campaign's directory, to write the figure into.
        table: The campaign's results table.
    """
    points = workload_points(table, "decode", "output_tokens")
    fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
    plot_with_fits(
        ax,
        points,
        SWEPT["decode"],
        lambda v: 1 / v,
        1,
        "decode J/token = a + b / concurrency",
    )
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter("{x:g}")
    ax.minorticks_off()
    ax.set_xlabel("Concurrency (requests)")
    ax.set_ylabel("Energy above idle per output token (J)")
    fig.savefig(campaign / "decode.png", dpi=200)


def plot_agentic(campaign: pathlib.Path, table: pd.DataFrame) -> None:
    """Plot agentic energy per turn against shared prefix length.

    Args:
        campaign: The campaign's directory, to write the figure into.
        table: The campaign's results table.
    """
    points = workload_points(table, "agentic", "requests")
    fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
    plot_with_fits(
        ax,
        points,
        SWEPT["agentic"],
        lambda v: v,
        1,
        "agentic J/turn = a + b * prefix",
    )
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Shared prefix length (tokens)")
    ax.set_ylabel("Energy above idle per turn (J)")
    fig.savefig(campaign / "agentic.png", dpi=200)


def plot_load(campaign: pathlib.Path, table: pd.DataFrame) -> None:
    """Plot the restart's power from off through idle, and print break-even.

    Args:
        campaign: The campaign's directory, to write the figure into.
        table: The campaign's results table.
    """
    rows = table[table.run.str.startswith("load-")].set_index("phase")
    run_dir = campaign / rows.experiment.iloc[0] / rows.run.iloc[0]
    phases = read(run_dir, "phases.csv")
    begins = dict(zip(phases.phase, phases.timestamp, strict=True))
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, layout="tight")
    for ax, (source, label) in zip(axes, SOURCES.items(), strict=True):
        trace = read(run_dir, f"{source}.csv")
        trace = trace[trace.timestamp.between(begins["off"], begins["end"])]
        seconds = (trace.timestamp - begins["load"]).dt.total_seconds()
        ax.plot(seconds, trace.power_w, linewidth=1)
        off_w = rows.loc["off", f"{source}_w"]
        idle_w = rows.loc["idle", f"{source}_w"]
        ax.axhline(off_w, color="k", linestyle=":", label="Off")
        ax.axhline(idle_w, color="k", linestyle="--", label="Idle")
        ax.set_ylabel(f"{label} power (W)")
        break_even_s = rows.loc["load", f"{source}_j_above"] / (idle_w - off_w)
        print(f"break-even idle  {source:<6}  {break_even_s:.0f} s")
    axes[0].legend()
    axes[1].set_xlabel("Time since docker start (s)")
    fig.savefig(campaign / "load.png", dpi=200)


def plot_run(
    ax: plt.Axes,
    run_dir: pathlib.Path,
    source: str,
    began: pd.Timestamp,
    color: str,
    label: str | None,
) -> None:
    """Plot one run's trace and shade its measured phases.

    Args:
        ax: The plot to draw on.
        run_dir: The run's directory.
        source: `"gpu"` or `"system"`.
        began: Time drawn as zero, in hours.
        color: Shade for the run's load or workload phase.
        label: Text along the bottom of the load or workload phase, if any.
    """
    phases = read(run_dir, "phases.csv")
    starts = (phases.timestamp - began).dt.total_seconds() / 3600
    # Settle is left blank.
    for phase, start, stop in zip(
        phases.phase, starts, starts.shift(-1), strict=True
    ):
        if phase in ["load", "workload"]:
            ax.axvspan(start, stop, color=color, alpha=0.3, linewidth=0)
            if label:
                ax.text(
                    (start + stop) / 2,
                    0.03,
                    label,
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )
        elif phase in ["off", "idle"]:
            ax.axvspan(start, stop, color="0.85", linewidth=0)
    trace = read(run_dir, f"{source}.csv")
    hours = (trace.timestamp - began).dt.total_seconds() / 3600
    ax.plot(hours, trace.power_w, color="k", linewidth=0.3)


def plot_timeline(
    campaign: pathlib.Path, table: pd.DataFrame, experiment: str
) -> None:
    """Plot both traces across an experiment's runs, with phases shaded.

    Args:
        campaign: The campaign's directory, to write the figure into.
        table: The campaign's results table.
        experiment: The experiment to plot.
    """
    table = table[table.experiment == experiment]
    experiment_dir = campaign / experiment
    # Run directories end in their start time, so this puts them in order.
    runs = sorted(table.run.unique(), key=lambda run: run.split("-")[1])
    names = sorted({run.split("-")[0] for run in runs})
    colors = {name: f"C{i}" for i, name in enumerate(names)}
    began = read(experiment_dir / runs[0], "phases.csv").timestamp.iloc[0]

    # Label each run with the value it swept, on the GPU row only.
    workloads = table[table.phase == "workload"].set_index("run")
    labels = {}
    for run in runs:
        name = run.split("-")[0]
        if name in SWEPT:
            labels[run] = f"{name} {workloads.loc[run, SWEPT[name]]:.0f}"
        else:
            labels[run] = name

    fig, axes = plt.subplots(
        2, 1, figsize=(20, 6), sharex=True, layout="tight"
    )
    for ax, (source, label) in zip(axes, SOURCES.items(), strict=True):
        for run in runs:
            text = labels[run] if source == "gpu" else None
            color = colors[run.split("-")[0]]
            plot_run(ax, experiment_dir / run, source, began, color, text)
        ax.set_ylabel(f"{label} (W)")
        ax.margins(x=0.005)
    axes[0].legend(
        handles=[
            patches.Patch(color=colors[name], alpha=0.3, label=name)
            for name in names
        ]
        + [patches.Patch(color="0.85", label="off or idle")],
        loc="lower right",
        bbox_to_anchor=(1, 1),
        ncols=len(names) + 1,
    )
    axes[1].set_xlabel("Hours since the experiment began")
    fig.savefig(campaign / f"{experiment}-timeline.png", dpi=200)


def main() -> None:
    """Write every figure into the campaign's directory and print the fits."""
    campaign = parse_args().campaign
    table = pd.read_csv(campaign / "results.csv")
    plot_prefill(campaign, table)
    plot_decode(campaign, table)
    plot_agentic(campaign, table)
    plot_load(campaign, table)
    for experiment in table.experiment.unique():
        plot_timeline(campaign, table, experiment)


if __name__ == "__main__":
    main()
