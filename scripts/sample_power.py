"""Run the hardware samplers, writing two CSVs into the output dir.

gpu.csv is the GPU's power trace, system.csv the whole server's. Both sample
at a fixed rate, 10 Hz and 1 Hz unless you pass the interval flags. Runs
until Ctrl-C unless --duration-s is given; pick your window by timestamp,
starting and stopping a second either side. Rerunning into a directory that
already holds CSVs truncates them.

Usage:
    uv run scripts/sample_power.py <out-dir> [--duration-s N]
"""

import argparse
import functools
import pathlib
import threading
import time

from power_experiments import record, samplers


def interval_ms(text: str, floor: int) -> int:
    """Parse a sampling period and reject anything under its hardware floor.

    Args:
        text: The flag's value, as typed.
        floor: Smallest period this hardware can honour, in ms.

    Returns:
        The period in ms.
    """
    value = int(text)
    if value < floor:
        raise argparse.ArgumentTypeError(
            f"must be at least {floor} ms, got {value}"
        )
    return value


def seconds(text: str) -> float:
    """Parse a positive number of seconds.

    Args:
        text: The flag's value, as typed.

    Returns:
        The duration in seconds.
    """
    value = float(text)
    if not value > 0:
        raise argparse.ArgumentTypeError(f"must be positive, got {value}")
    return value


def parse_args() -> argparse.Namespace:
    """Parse the command line.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "out_dir",
        type=pathlib.Path,
        help="directory to write gpu.csv and system.csv into",
    )
    parser.add_argument(
        "--gpu-interval-ms",
        type=functools.partial(interval_ms, floor=samplers.GPU_FLOOR_MS),
        default=samplers.GPU_FLOOR_MS,
        help="sampling period for gpu.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--system-interval-ms",
        type=functools.partial(interval_ms, floor=samplers.SYSTEM_FLOOR_MS),
        default=samplers.SYSTEM_FLOOR_MS,
        help="sampling period for system.csv (default: %(default)s)",
    )
    parser.add_argument(
        "--duration-s",
        type=seconds,
        default=None,
        help="stop after this many seconds instead of waiting for Ctrl-C",
    )
    return parser.parse_args()


def main() -> None:
    """Start both samplers, wait for the deadline or Ctrl-C, then report."""
    # Read the flags and set up the run.
    args = parse_args()
    start = time.monotonic()
    deadline = None if args.duration_s is None else start + args.duration_s
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Start one background thread per CSV.
    streams = {
        "gpu.csv": samplers.gpu_samples(args.gpu_interval_ms),
        "system.csv": samplers.system_samples(args.system_interval_ms),
    }
    writers = {}
    for filename, samples in streams.items():
        writer = threading.Thread(
            target=record.write_csv,
            args=(args.out_dir / filename, samples),
            daemon=True,
        )
        writer.start()
        writers[filename] = writer

    # Say where we are writing and for how long.
    stop_at = (
        "until Ctrl-C" if args.duration_s is None else f"{args.duration_s} s"
    )
    print(
        f"writing to {args.out_dir}/ at gpu {args.gpu_interval_ms} ms,"
        f" system {args.system_interval_ms} ms - {stop_at}",
        flush=True,
    )

    # Keep the process alive until the deadline or Ctrl-C.
    poll_s = min(args.gpu_interval_ms, args.system_interval_ms) / 1000
    try:
        while deadline is None or time.monotonic() < deadline:
            time.sleep(poll_s)
            for filename, writer in writers.items():
                if not writer.is_alive():
                    raise SystemExit(f"{filename} writer died, run is void")
    except KeyboardInterrupt:
        pass

    # Report what was written.
    print(f"stopped after {time.monotonic() - start:.1f} s", flush=True)
    for filename in writers:
        with open(args.out_dir / filename) as f:
            rows = len(f.readlines()) - 1
        print(f"  {filename}  {rows} rows", flush=True)


if __name__ == "__main__":
    main()
