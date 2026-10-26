"""Write CSVs whose rows each carry a wall-clock UTC timestamp."""

import csv
import datetime
import itertools
import pathlib
from collections.abc import Iterator


def now() -> str:
    """Read the wall clock as a CSV timestamp.

    Returns:
        The current time in UTC, ISO 8601 to the microsecond.
    """
    return datetime.datetime.now(datetime.UTC).isoformat(
        timespec="microseconds"
    )


def write_csv(path: pathlib.Path, samples: Iterator[dict]) -> None:
    """Consume samples forever, writing each row with a timestamp.

    Args:
        path: File to write.
        samples: Endless generator of dicts, whose keys become the columns.
    """
    with open(path, "w", newline="") as f:
        # Peek at one row for the columns, then write it with the rest.
        first = next(samples)
        writer = csv.DictWriter(f, fieldnames=["timestamp", *first])
        writer.writeheader()
        f.flush()
        for row in itertools.chain([first], samples):
            writer.writerow({"timestamp": now(), **row})
            # The thread dies at exit, so flush every row.
            f.flush()


def write_phase(path: pathlib.Path, phase: str) -> None:
    """Append a row saying a phase begins now.

    Args:
        path: File to append to, created with its header if absent.
        phase: Name of the phase.
    """
    new = not path.exists()
    with open(path, "a") as f:
        if new:
            f.write("timestamp,phase\n")
        f.write(f"{now()},{phase}\n")
