"""Record one named run from a config file into results/<name>-<time>/.

Run from the repo root.

Usage:
    uv run scripts/run.py <config> <name>
"""

import argparse
import datetime
import pathlib
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
    parser.add_argument("config", type=pathlib.Path, help="YAML config file")
    parser.add_argument(
        "name", help="which run in the config, e.g. load, prefill, decode"
    )
    return parser.parse_args()


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


def main() -> None:
    """Record the run's phases and traces into its own directory."""
    # Read the config, merge the run over the shared settings, and save it.
    args = parse_args()
    config = yaml.safe_load(args.config.read_text())
    shared = {k: v for k, v in config.items() if k != "runs"}
    run = {**shared, **config["runs"][args.name]}
    began = datetime.datetime.now(datetime.UTC)
    out_dir = pathlib.Path("results") / f"{args.name}-{began:%Y%m%dT%H%M%S}"
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
        # docker stop returns once the container is down.
        subprocess.run(["docker", "stop", run["container"]], check=True)
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


if __name__ == "__main__":
    main()
