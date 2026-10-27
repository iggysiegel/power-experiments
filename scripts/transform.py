"""Tabulate every complete run in results/ into results/results.csv.

Run from the repo root.

Usage:
    uv run scripts/transform.py
"""

import json
import pathlib
import sys

import pandas as pd
import yaml

from power_experiments import energy

RESULTS = pathlib.Path("results")


def workload_columns(run_dir: pathlib.Path) -> dict:
    """Read what a run's workload did and what it was asked to do.

    Args:
        run_dir: The run's directory.

    Returns:
        Requests and output tokens as completed, then every bench flag as
        set.
    """
    bench_flags = yaml.safe_load((run_dir / "config.yml").read_text())["bench"]
    bench = json.loads((run_dir / "bench.json").read_text())
    return {
        "requests": bench["completed"],
        "output_tokens": bench["total_output_tokens"],
        **bench_flags,
    }


def phase_rows(run_dir: pathlib.Path, phases: pd.DataFrame) -> list[dict]:
    """Measure each phase of one run on both power traces.

    Args:
        run_dir: The run's directory.
        phases: The run's phases.csv, ending with `end`.

    Returns:
        One row per measured phase, keyed by column name.
    """
    # Each phase ends when the next begins; settle and end are not measured.
    phases["stop"] = phases.timestamp.shift(-1)
    phases = phases[~phases.phase.isin(["settle", "end"])]

    # Load is measured above off, and workload above idle.
    above_phase = {"load": "off", "workload": "idle"}

    rows = []
    watts = {}  # (phase, source) -> mean W, for later phases' baselines
    for phase, start, stop in zip(
        phases.phase, phases.timestamp, phases.stop, strict=True
    ):
        above = above_phase.get(phase)
        row = {
            "run": run_dir.name,
            "phase": phase,
            "seconds": (stop - start).total_seconds(),
        }
        for source in ["gpu", "system"]:
            baseline_w = watts[above, source] if above else 0.0
            window = energy.window_energy(
                run_dir / f"{source}.csv", start, stop, source, baseline_w
            )
            watts[phase, source] = window["mean_w"]
            row[f"{source}_w"] = window["mean_w"]
            row[f"{source}_j"] = window["joules"]
            row[f"{source}_j_above"] = (
                window["joules_incremental"] if above else None
            )
        if phase == "workload":
            row.update(workload_columns(run_dir))
        rows.append(row)
    return rows


def main() -> None:
    """Write one row per measured phase of every complete run."""
    rows = []
    for run_dir in sorted(RESULTS.glob("*/")):
        phases = pd.read_csv(run_dir / "phases.csv", parse_dates=["timestamp"])
        if phases.phase.iloc[-1] != "end":
            print(f"skipping incomplete run {run_dir.name}", file=sys.stderr)
            continue
        rows.extend(phase_rows(run_dir, phases))

    # Integer columns stay integers despite the blanks on non-workload rows.
    table = pd.DataFrame(rows).round(1).convert_dtypes()
    table.to_csv(RESULTS / "results.csv", index=False)
    print(f"wrote {len(table)} rows to {RESULTS / 'results.csv'}")


if __name__ == "__main__":
    main()
