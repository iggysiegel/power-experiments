"""Guard the joule arithmetic against silent unit and weighting errors."""

import datetime

from power_experiments import energy

### Test Setup ###

START = datetime.datetime(2026, 1, 1, tzinfo=datetime.UTC)


def at(seconds):
    """A timestamp this many seconds into the test window."""
    return START + datetime.timedelta(seconds=seconds)


def write_csv(path, header, rows):
    """Write a sampler-shaped CSV with the given header and rows."""
    path.write_text("\n".join([header, *rows]) + "\n")
    return path


### Unit Tests ###


def test_window_energy_prefers_the_counter(tmp_path):
    """Read joules from the counter, not by integrating the power column."""
    # 100 W for 10 s integrates to 1000 J; the counter says 1010. Anything
    # that integrates instead of reading the counter returns 1000 and fails.
    path = write_csv(
        tmp_path / "gpu.csv",
        "timestamp,power_w,energy_mj",
        [
            "2026-01-01T00:00:00+00:00,100,5000000",
            "2026-01-01T00:00:05+00:00,100,5500000",
            "2026-01-01T00:00:10+00:00,100,6010000",
        ],
    )

    got = energy.window_energy(path, START, at(10), "gpu")

    assert got["joules"] == 1010.0


def test_window_energy_integrates_without_a_counter(tmp_path):
    """Fall back to the trapezoid when the file has no counter."""
    # 100 W for 2 s then a ramp to 200 W over 8 s: 200 + 1200 = 1400 J.
    # Uneven spacing makes mean_w energy-weighted (140 W) rather than the
    # arithmetic mean (133 W).
    path = write_csv(
        tmp_path / "system.csv",
        "timestamp,power_w",
        [
            "2026-01-01T00:00:00+00:00,100",
            "2026-01-01T00:00:02+00:00,100",
            "2026-01-01T00:00:10+00:00,200",
        ],
    )

    got = energy.window_energy(path, START, at(10), "system", baseline_w=50)

    assert got["joules"] == 1400.0
    assert got["mean_w"] == 140.0
    assert got["joules_incremental"] == 900.0
