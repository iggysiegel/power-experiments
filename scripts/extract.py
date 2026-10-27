"""Record every run of an experiment file into results/<name>-<time>/.

Run from the repo root.

Usage:
    uv run scripts/extract.py <experiment>
"""

import argparse
import datetime
import multiprocessing
import pathlib
import random
import subprocess
import threading
import time
import urllib.request

import yaml

from power_experiments import record, samplers


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "experiment", type=pathlib.Path, help="YAML experiment file"
    )
    return parser.parse_args()


def expand_points(config: dict) -> list[dict]:
    """Expand each run in an experiment into one set of settings per point.

    Args:
        config: The experiment file's contents.

    Returns:
        Each point's settings: the shared settings, the run's, and its name,
        with that point's sweep values in `bench`.
    """
    shared = {k: v for k, v in config.items() if k != "runs"}
    points = []
    for name, run in config["runs"].items():
        settings = {**shared, **run, "name": name}
        sweep = settings.pop("sweep", None)
        if not sweep:
            points.append(settings)
            continue
        for values in zip(*sweep.values(), strict=True):
            sweep_flags = dict(zip(sweep, values, strict=True))
            bench = {**settings["bench"], **sweep_flags}
            points.append({**settings, "bench": bench})
    return points


def wait_until_rested(max_c: float) -> None:
    """Wait until the GPU is at or below a temperature.

    Args:
        max_c: Highest temperature that counts as resting, in °C.
    """
    while True:
        temp = next(samplers.gpu_samples())["temp_c"]
        if temp <= max_c:
            return
        print(f"GPU at {temp} °C, waiting for {max_c} °C", flush=True)
        time.sleep(10)


def vllm_bench_command(run: dict, out_dir: pathlib.Path) -> list[str]:
    """Build the vllm-bench command line for a run's workload.

    Args:
        run: The run's settings, with `url`, `model`, and `bench`.
        out_dir: Directory for vllm-bench to save bench.json into.

    Returns:
        The command, one word per item.
    """
    flags = {
        "base-url": run["url"],
        "model": run["model"],
        "save-result": True,
        "result-dir": out_dir,
        "result-filename": "bench.json",
        **run["bench"],
    }
    command = ["vllm-bench"]
    for flag, value in flags.items():
        command.append(f"--{flag}")
        if value is not True:
            command.append(str(value))
    return command


def record_run(run: dict) -> None:
    """Record one run's phases and traces into its own directory.

    Args:
        run: The run's settings, as one point from `expand_points`.
    """
    # A fresh seed keeps prompts from repeating across runs, so the prefix
    # cache only helps within a run. Save the settings with it.
    began = datetime.datetime.now(datetime.UTC)
    if "bench" in run:
        run["bench"]["seed"] = int(began.timestamp())
    out_dir = pathlib.Path("results") / f"{run['name']}-{began:%Y%m%dT%H%M%S}"
    out_dir.mkdir(parents=True)
    (out_dir / "config.yml").write_text(yaml.safe_dump(run, sort_keys=False))
    print(f"writing to {out_dir}/", flush=True)

    # Start one background thread per CSV; they end with the process.
    streams = {
        "gpu.csv": samplers.gpu_samples(),
        "system.csv": samplers.system_samples(),
    }
    for filename, samples in streams.items():
        threading.Thread(
            target=record.write_csv,
            args=(out_dir / filename, samples),
            daemon=True,
        ).start()

    phases = out_dir / "phases.csv"
    if run.get("restart"):
        subprocess.run(["docker", "stop", run["container"]], check=True)
        record.write_phase(phases, "settle")
        print("settle", flush=True)
        wait_until_rested(run["rest_max_c"])

        record.write_phase(phases, "off")
        print("off", flush=True)
        time.sleep(run["off_s"])

        # Load lasts until the server answers.
        record.write_phase(phases, "load")
        print("load", flush=True)
        subprocess.run(["docker", "start", run["container"]], check=True)
        while True:
            try:
                urllib.request.urlopen(f"{run['url']}/v1/models", timeout=1)
                break
            except OSError:
                time.sleep(1)

    record.write_phase(phases, "settle")
    print("settle", flush=True)
    wait_until_rested(run["rest_max_c"])

    record.write_phase(phases, "idle")
    print("idle", flush=True)
    time.sleep(run["idle_s"])

    if "bench" in run:
        record.write_phase(phases, "workload")
        print("workload", flush=True)
        subprocess.run(vllm_bench_command(run, out_dir), check=True)

    record.write_phase(phases, "end")
    print("end", flush=True)

    if run.get("restart"):
        # Keep vLLM's own timestamped account of the restart.
        since = str(began.timestamp())
        command = ["docker", "logs", "--timestamps", "--since", since]
        with open(out_dir / "container.log", "w") as log:
            subprocess.run(
                [*command, run["container"]],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )


def main() -> None:
    """Record every point of every run, in shuffled order."""
    args = parse_args()
    config = yaml.safe_load(args.experiment.read_text())
    points = expand_points(config)
    # Shuffled so drift over the night does not line up with a sweep.
    random.shuffle(points)
    for run in points:
        # Each run in its own process, since its sampler threads only stop
        # when the process exits.
        process = multiprocessing.Process(target=record_run, args=(run,))
        process.start()
        process.join()
        if process.exitcode != 0:
            print(f"{run['name']} {run.get('bench')} failed", flush=True)


if __name__ == "__main__":
    main()
