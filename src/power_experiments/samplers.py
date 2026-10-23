"""Sample GPU and whole-server power from the hardware.

Each sampler is an endless generator of dicts, one row per yield. Timing,
writing, and stopping are the caller's job.
"""

import subprocess
import sys
import time
from collections.abc import Iterator

import pynvml

# Below this, GPU readings repeat.
GPU_FLOOR_MS = 100

# Below this, DCMI repeats.
SYSTEM_FLOOR_MS = 1000


def gpu_samples(
    interval_ms: int = GPU_FLOOR_MS,
) -> Iterator[dict[str, float | int]]:
    """Sample the GPU's power, energy, and health at a fixed rate.

    Args:
        interval_ms: Period between rows.

    Yields:
        One row per interval. `energy_mj` is a lifetime total, so only the
        difference between two rows is meaningful.
    """
    pynvml.nvmlInit()
    # One GPU, so index 0 is unambiguous.
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    while True:
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        # One row is handed off and forgotten, so a long run costs no memory.
        yield {
            "power_w": pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
            "energy_mj": pynvml.nvmlDeviceGetTotalEnergyConsumption(handle),
            "util_gpu_pct": util.gpu,
            "util_mem_pct": util.memory,
            "mem_used_mib": mem.used // 2**20,
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(
                handle, pynvml.NVML_CLOCK_SM
            ),
            "temp_c": pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            ),
        }
        time.sleep(interval_ms / 1000)


def system_samples(
    interval_ms: int = SYSTEM_FLOOR_MS,
) -> Iterator[dict[str, int]]:
    """Sample whole-server power at a fixed rate.

    Args:
        interval_ms: Period between rows.

    Yields:
        One row per interval with `power_w` for the whole server.
    """
    while True:
        cmd = ["sudo", "-n", "ipmitool", "-I", "open"]
        try:
            out = subprocess.run(
                [*cmd, "dcmi", "power", "reading"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            line = next(
                r
                for r in out.splitlines()
                if "Instantaneous power reading" in r
            )
            yield {"power_w": int(line.split(":")[1].split()[0])}
        # One bad poll does not end the run.
        except subprocess.CalledProcessError as exc:
            print(f"ipmitool failed: {exc}", file=sys.stderr, flush=True)
        time.sleep(interval_ms / 1000)
