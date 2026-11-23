"""Plot each workload's energy curve per model, each model's power during the
restart, and power and each run's energy across each
experiment, into a campaign's `figures` directory.

Usage:
    uv run scripts/plot.py <campaign directory>
"""

import argparse
import pathlib
import sys
from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import lines, patches, ticker

SOURCES = {"gpu": "GPU", "system": "Whole server"}

# Shades for an experiment's runs: the off or idle level in grey, and each
# load or workload phase in its run's colour at this opacity.
LEVEL_SHADE = "0.85"
PHASE_ALPHA = 0.3

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


def save(fig: plt.Figure, campaign: pathlib.Path, name: str) -> None:
    """Write a figure into the campaign's `figures` directory.

    Args:
        fig: The figure to write.
        campaign: The campaign's directory.
        name: The figure's file name, such as `prefill.png`.
    """
    figures = campaign / "figures"
    figures.mkdir(exist_ok=True)
    fig.savefig(figures / name, dpi=200)


def read(run_dir: pathlib.Path, filename: str) -> pd.DataFrame:
    """Read one of a run's timestamped CSVs.

    Args:
        run_dir: The run's directory.
        filename: Such as `phases.csv` or `gpu.csv`.

    Returns:
        The CSV, with `timestamp` parsed.
    """
    return pd.read_csv(run_dir / filename, parse_dates=["timestamp"])


def run_kind(run: str) -> str:
    """Name what a run measured.

    Args:
        run: The run's directory name, such as `prefill-20261027T005450`.

    Returns:
        Its workload, such as `prefill`, or `load`.
    """
    return run.split("-")[0]


def model_colors(table: pd.DataFrame) -> dict[str, str]:
    """Give each experiment, and so each model, its own colour.

    Args:
        table: The campaign's results table.

    Returns:
        Each experiment, in alphabetical order, mapped to its colour.
    """
    experiments = sorted(table.experiment.unique())
    return {experiment: f"C{i}" for i, experiment in enumerate(experiments)}


def workload_points(table: pd.DataFrame, name: str, per: str) -> pd.DataFrame:
    """Select one workload's runs and divide their energy by a count.

    Args:
        table: The results table.
        name: The workload, such as `prefill`.
        per: Column to divide joules above idle by.

    Returns:
        The workload rows, with `gpu` and `system` holding joules above idle
        per unit of `per`.
    """
    is_workload = table.phase == "workload"
    rows = table[is_workload & (table.run.map(run_kind) == name)].copy()
    for source in SOURCES:
        rows[source] = rows[f"{source}_j_above"] / rows[per]
    return rows


def with_context_fill(points: pd.DataFrame, tokens: pd.Series) -> pd.DataFrame:
    """Add each point's context fill, leaving out models with no window.

    Args:
        points: Rows from `workload_points`.
        tokens: Tokens in context at each point's request.

    Returns:
        The points of experiments that state a context window, with `fill`
        holding `tokens` as a fraction of it.
    """
    # The pilot ran before experiments stated their context window.
    unknown = points.context_window.isna()
    if unknown.any():
        left_out = ", ".join(sorted(points[unknown].experiment.unique()))
        print(
            f"no context window, so no context fill: {left_out}",
            file=sys.stderr,
        )
    points = points[~unknown].copy()
    points["fill"] = tokens[~unknown] / points.context_window
    return points


def plot_models(
    points: pd.DataFrame, x: str, colors: dict[str, str]
) -> tuple[plt.Figure, np.ndarray]:
    """Plot each model's points joined in order, one panel per source.

    Args:
        points: Rows from `workload_points`.
        x: Column holding the x-axis value.
        colors: From `model_colors`.

    Returns:
        The figure and its GPU and whole-server panels.
    """
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.5), sharey=True, layout="tight"
    )
    for experiment, rows in points.groupby("experiment"):
        rows = rows.sort_values(x)
        for ax, source in zip(axes, SOURCES, strict=True):
            ax.plot(
                rows[x],
                rows[source],
                "o-",
                color=colors[experiment],
                linewidth=1,
                label=experiment,
            )
    for ax, label in zip(axes, SOURCES.values(), strict=True):
        ax.set_title(label)
    axes[0].legend()
    return fig, axes


def count_x(ax: plt.Axes, values: pd.Series) -> None:
    """Draw a count on a log2 axis, ticked at powers of two.

    Args:
        ax: The plot whose x-axis to set.
        values: The counts drawn, to span with ticks.
    """
    ax.set_xscale("log", base=2)
    # Round the ends so near-full points such as 253,952 get a tick nearby.
    lo, hi = np.round(np.log2([values.min(), values.max()])).astype(int)
    ticks = 2 ** np.arange(lo, hi + 1)
    ax.set_xticks(
        ticks, [f"{t // 1024}k" if t >= 1024 else str(t) for t in ticks]
    )
    ax.minorticks_off()


def fill_x(ax: plt.Axes) -> None:
    """Draw context fill on a log axis, as percentages.

    Args:
        ax: The plot whose x-axis to set.
    """
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ax.xaxis.set_major_formatter(lambda v, _: f"{100 * v:g}%")
    ax.minorticks_off()
    ax.set_xlabel("Context fill (of the context window)")


def log_y(ax: plt.Axes) -> None:
    """Draw energy on a log axis, ticked at 1, 2, and 5 of each decade.

    Args:
        ax: The plot whose y-axis to set.
    """
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(ticker.LogLocator(subs=(1, 2, 5)))
    ax.yaxis.set_major_formatter("{x:g}")
    ax.minorticks_off()


def plot_prefill(
    campaign: pathlib.Path, table: pd.DataFrame, colors: dict[str, str]
) -> None:
    """Plot prefill energy per request against input length and fill.

    Args:
        campaign: The campaign's directory.
        table: The campaign's results table.
        colors: From `model_colors`.
    """
    points = workload_points(table, "prefill", "requests")
    x = SWEPT["prefill"]
    fig, axes = plot_models(points, x, colors)
    for ax in axes:
        log_y(ax)
        count_x(ax, points[x])
        ax.set_xlabel("Input length (tokens)")
    axes[0].set_ylabel("Energy above idle per request (J)")
    save(fig, campaign, "prefill.png")

    points = with_context_fill(points, points[x])
    if points.empty:
        return
    fig, axes = plot_models(points, "fill", colors)
    for ax in axes:
        log_y(ax)
        fill_x(ax)
    axes[0].set_ylabel("Energy above idle per request (J)")
    save(fig, campaign, "prefill-fill.png")


def plot_decode(
    campaign: pathlib.Path, table: pd.DataFrame, colors: dict[str, str]
) -> None:
    """Plot decode energy per output token against concurrency.

    Args:
        campaign: The campaign's directory.
        table: The campaign's results table.
        colors: From `model_colors`.
    """
    points = workload_points(table, "decode", "output_tokens")
    x = SWEPT["decode"]
    fig, axes = plot_models(points, x, colors)
    for ax in axes:
        log_y(ax)
        count_x(ax, points[x])
        ax.set_xlabel("Concurrency (requests)")
    axes[0].set_ylabel("Energy above idle per output token (J)")
    save(fig, campaign, "decode.png")


def plot_agentic(
    campaign: pathlib.Path, table: pd.DataFrame, colors: dict[str, str]
) -> None:
    """Plot agentic energy per turn against shared prefix length and fill.

    Args:
        campaign: The campaign's directory.
        table: The campaign's results table.
        colors: From `model_colors`.
    """
    points = workload_points(table, "agentic", "requests")
    x = SWEPT["agentic"]
    fig, axes = plot_models(points, x, colors)
    for ax in axes:
        ax.set_ylim(bottom=0)
        count_x(ax, points[x])
        ax.set_xlabel("Shared prefix length (tokens)")
    axes[0].set_ylabel("Energy above idle per turn (J)")
    save(fig, campaign, "agentic.png")

    # The bench counts each turn's prompt as the prefix plus one input and
    # one output per turn so far, its own included: on the pilot, the final
    # turn's prompt was the prefix plus 2,048 tokens for 8 turns of 128.
    turns = points["multi-turn-num-turns"]
    per_turn = points["random-input-len"] + points["random-output-len"]
    final_prompt = points[x] + turns * per_turn
    points = with_context_fill(points, final_prompt)
    if points.empty:
        return
    fig, axes = plot_models(points, "fill", colors)
    for ax in axes:
        ax.set_ylim(bottom=0)
        fill_x(ax)
    axes[0].set_ylabel("Energy above idle per turn (J)")
    save(fig, campaign, "agentic-fill.png")


def plot_load(
    campaign: pathlib.Path, table: pd.DataFrame, colors: dict[str, str]
) -> None:
    """Plot each model's restart power, and print its break-even time.

    Args:
        campaign: The campaign's directory.
        table: The campaign's results table.
        colors: From `model_colors`.
    """
    loads = table[table.run.map(run_kind) == "load"]
    width = loads.experiment.str.len().max()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, layout="tight")
    for experiment, rows in loads.groupby("experiment"):
        color = colors[experiment]
        rows = rows.set_index("phase")
        run_dir = campaign / experiment / rows.run.iloc[0]
        phases = read(run_dir, "phases.csv")
        begins = dict(zip(phases.phase, phases.timestamp, strict=True))
        for ax, source in zip(axes, SOURCES, strict=True):
            trace = read(run_dir, f"{source}.csv")
            trace = trace[
                trace.timestamp.between(begins["off"], begins["end"])
            ]
            seconds = (trace.timestamp - begins["load"]).dt.total_seconds()
            ax.plot(seconds, trace.power_w, color=color, linewidth=1)
            off_w = rows.loc["off", f"{source}_w"]
            idle_w = rows.loc["idle", f"{source}_w"]
            ax.axhline(off_w, color=color, linestyle=":", linewidth=1)
            ax.axhline(idle_w, color=color, linestyle="--", linewidth=1)
            load_j = rows.loc["load", f"{source}_j_above"]
            break_even_s = load_j / (idle_w - off_w)
            print(
                f"break-even idle  {experiment:<{width}}  {source:<6}"
                f"  {break_even_s:.0f} s"
            )
    for ax, label in zip(axes, SOURCES.values(), strict=True):
        ax.set_ylabel(f"{label} power (W)")
    axes[0].legend(
        handles=[
            lines.Line2D([], [], color=color, label=experiment)
            for experiment, color in colors.items()
            if experiment in set(loads.experiment)
        ]
        + [
            lines.Line2D([], [], color="k", linestyle=":", label="Off"),
            lines.Line2D([], [], color="k", linestyle="--", label="Idle"),
        ]
    )
    axes[1].set_xlabel("Time since docker start (s)")
    save(fig, campaign, "load.png")


def run_labels(table: pd.DataFrame) -> dict[str, str]:
    """Order one experiment's runs by start time and label each.

    Args:
        table: The results table's rows for one experiment.

    Returns:
        Each run's name, in the order it ran, mapped to its workload and the
        value it swept, or to just its name for a load run.
    """
    # Run directories end in their start time, so this puts them in order.
    runs = sorted(table.run.unique(), key=lambda run: run.split("-")[1])
    workloads = table[table.phase == "workload"].set_index("run")
    labels = {}
    for run in runs:
        kind = run_kind(run)
        if kind in SWEPT:
            labels[run] = f"{kind} {workloads.loc[run, SWEPT[kind]]:.0f}"
        else:
            labels[run] = kind
    return labels


def run_colors(runs: Iterable[str]) -> dict[str, str]:
    """Give each workload, and load, its own colour.

    Args:
        runs: Run directory names, such as `prefill-20261027T005450`.

    Returns:
        Each of `run_kind`'s names among the runs, in alphabetical order,
        mapped to its colour.
    """
    kinds = sorted({run_kind(run) for run in runs})
    return {kind: f"C{i}" for i, kind in enumerate(kinds)}


def legend_runs(ax: plt.Axes, colors: dict[str, str]) -> None:
    """Key the shades of an experiment's runs, above a plot.

    Args:
        ax: The plot to put the legend over.
        colors: From `run_colors`.
    """
    ax.legend(
        handles=[
            patches.Patch(color=color, alpha=PHASE_ALPHA, label=kind)
            for kind, color in colors.items()
        ]
        + [patches.Patch(color=LEVEL_SHADE, label="off or idle")],
        loc="lower right",
        bbox_to_anchor=(1, 1),
        ncols=len(colors) + 1,
    )


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
            ax.axvspan(
                start, stop, color=color, alpha=PHASE_ALPHA, linewidth=0
            )
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
            ax.axvspan(start, stop, color=LEVEL_SHADE, linewidth=0)
    trace = read(run_dir, f"{source}.csv")
    hours = (trace.timestamp - began).dt.total_seconds() / 3600
    ax.plot(hours, trace.power_w, color="k", linewidth=0.3)


def plot_timeline(
    campaign: pathlib.Path, table: pd.DataFrame, experiment: str
) -> None:
    """Plot both traces across an experiment's runs, with phases shaded.

    Args:
        campaign: The campaign's directory.
        table: The campaign's results table.
        experiment: The experiment to plot.
    """
    table = table[table.experiment == experiment]
    experiment_dir = campaign / experiment
    labels = run_labels(table)
    colors = run_colors(labels)
    first = next(iter(labels))
    began = read(experiment_dir / first, "phases.csv").timestamp.iloc[0]

    fig, axes = plt.subplots(
        2, 1, figsize=(20, 6), sharex=True, layout="tight"
    )
    for ax, (source, label) in zip(axes, SOURCES.items(), strict=True):
        for run in labels:
            # Label each run on the GPU row only.
            text = labels[run] if source == "gpu" else None
            color = colors[run_kind(run)]
            plot_run(ax, experiment_dir / run, source, began, color, text)
        ax.set_ylabel(f"{label} (W)")
        ax.margins(x=0.005)
    legend_runs(axes[0], colors)
    axes[1].set_xlabel("Hours since the experiment began")
    save(fig, campaign, f"{experiment}-timeline.png")


def plot_run_energy(
    campaign: pathlib.Path, table: pd.DataFrame, experiment: str
) -> None:
    """Plot each run's energy, split at its idle or off level, in run order.

    Args:
        campaign: The campaign's directory.
        table: The campaign's results table.
        experiment: The experiment to plot.
    """
    table = table[table.experiment == experiment]
    labels = run_labels(table)
    colors = run_colors(labels)
    # Each run's load or workload phase, which is measured above a level.
    rows = table[table.phase.isin(["load", "workload"])].set_index("run")
    rows = rows.loc[list(labels)]
    x = np.arange(len(rows))
    shades = [colors[run_kind(run)] for run in rows.index]

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, layout="tight"
    )
    for ax, (source, label) in zip(axes, SOURCES.items(), strict=True):
        total_kj = rows[f"{source}_j"] / 1000
        above_kj = rows[f"{source}_j_above"] / 1000
        # What the run's idle level (off for load) would have drawn over the
        # phase; taken as the difference so the bar's total is the phase's
        # energy exactly, despite the table's rounded watts.
        level_kj = total_kj - above_kj
        ax.bar(x, level_kj, color=LEVEL_SHADE)
        ax.bar(x, above_kj, bottom=level_kj, color=shades, alpha=PHASE_ALPHA)
        ax.set_ylabel(f"{label} energy (kJ)")
        ax.grid(axis="y", color="0.9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.margins(x=0.01)
    legend_runs(axes[0], colors)
    axes[1].set_xticks(x, list(labels.values()), rotation=90, fontsize=7)
    save(fig, campaign, f"{experiment}-run-energy.png")


def main() -> None:
    """Write every figure into the campaign's figures and print break-even."""
    campaign = parse_args().campaign
    table = pd.read_csv(campaign / "results.csv")
    colors = model_colors(table)
    plot_prefill(campaign, table, colors)
    plot_decode(campaign, table, colors)
    plot_agentic(campaign, table, colors)
    plot_load(campaign, table, colors)
    for experiment in table.experiment.unique():
        plot_timeline(campaign, table, experiment)
        plot_run_energy(campaign, table, experiment)


if __name__ == "__main__":
    main()
