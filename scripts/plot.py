"""Plot each workload's energy curve with its fit, and power during restarts.

Run from the repo root.

Usage:
    uv run scripts/plot.py
"""

import pathlib
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import ticker
from numpy.polynomial import polynomial

RESULTS = pathlib.Path("results")
SOURCES = {"gpu": "GPU", "system": "Whole server"}


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
    """Plot each source's repeats and polynomial fit, and print the fits.

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
            label=f"{label} fit",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(swept, [str(int(v)) for v in swept])
    ax.minorticks_off()
    ax.legend()


def plot_prefill(table: pd.DataFrame) -> None:
    """Plot prefill energy per request against input length.

    Args:
        table: The results table.
    """
    points = workload_points(table, "prefill", "requests")
    fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
    plot_with_fits(
        ax,
        points,
        "random-input-len",
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
    fig.savefig(RESULTS / "prefill.png", dpi=200)


def plot_decode(table: pd.DataFrame) -> None:
    """Plot decode energy per output token against concurrency.

    Args:
        table: The results table.
    """
    points = workload_points(table, "decode", "output_tokens")
    fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
    plot_with_fits(
        ax,
        points,
        "max-concurrency",
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
    fig.savefig(RESULTS / "decode.png", dpi=200)


def plot_agentic(table: pd.DataFrame) -> None:
    """Plot agentic energy per turn against shared prefix length.

    Args:
        table: The results table.
    """
    points = workload_points(table, "agentic", "requests")
    fig, ax = plt.subplots(figsize=(6, 4), layout="tight")
    plot_with_fits(
        ax,
        points,
        "random-prefix-len",
        lambda v: v,
        1,
        "agentic J/turn = a + b * prefix",
    )
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Shared prefix length (tokens)")
    ax.set_ylabel("Energy above idle per turn (J)")
    fig.savefig(RESULTS / "agentic.png", dpi=200)


def plot_load(table: pd.DataFrame) -> None:
    """Plot each restart's power from off through idle, and print break-even.

    Args:
        table: The results table.
    """
    rows = table[table.run.str.startswith("load-")]
    level = rows.groupby("phase").mean(numeric_only=True)
    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True, layout="tight")
    for ax, (source, label) in zip(axes, SOURCES.items(), strict=True):
        for i, run in enumerate(rows.run.unique()):
            phases = pd.read_csv(
                RESULTS / run / "phases.csv", parse_dates=["timestamp"]
            )
            begins = dict(zip(phases.phase, phases.timestamp, strict=True))
            trace = pd.read_csv(
                RESULTS / run / f"{source}.csv", parse_dates=["timestamp"]
            )
            trace = trace[
                trace.timestamp.between(begins["off"], begins["end"])
            ]
            seconds = (trace.timestamp - begins["load"]).dt.total_seconds()
            ax.plot(
                seconds, trace.power_w, linewidth=1, label=f"Restart {i + 1}"
            )
        off_w = level.loc["off", f"{source}_w"]
        idle_w = level.loc["idle", f"{source}_w"]
        ax.axhline(off_w, color="k", linestyle=":", label="Off (mean)")
        ax.axhline(idle_w, color="k", linestyle="--", label="Idle (mean)")
        ax.set_ylabel(f"{label} power (W)")
        break_even_s = level.loc["load", f"{source}_j_above"] / (
            idle_w - off_w
        )
        print(f"break-even idle  {source:<6}  {break_even_s:.0f} s")
    axes[0].legend()
    axes[1].set_xlabel("Time since docker start (s)")
    fig.savefig(RESULTS / "load.png", dpi=200)


def main() -> None:
    """Write every figure into results/ and print the fits."""
    table = pd.read_csv(RESULTS / "results.csv")
    plot_prefill(table)
    plot_decode(table)
    plot_agentic(table)
    plot_load(table)


if __name__ == "__main__":
    main()
