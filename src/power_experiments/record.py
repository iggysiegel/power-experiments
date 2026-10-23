"""Write sampler rows to CSV, one wall-clock UTC timestamp per row.

Runs until its thread is killed, so the file on disk is the only record.
"""

import csv
import datetime
import itertools
import pathlib
from collections.abc import Iterator


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
            stamp = datetime.datetime.now(datetime.UTC).isoformat(
                timespec="microseconds"
            )
            writer.writerow({"timestamp": stamp, **row})
            # The thread dies at exit, so flush every row.
            f.flush()
