"""Guard the parsing of a format we do not control."""

from types import SimpleNamespace

from power_experiments import samplers

### Test Setup ###

# Real output from `ipmitool dcmi power reading` on this BMC. Note the average
# reads 261 W no matter what the server is doing, which is why we ignore it.
DCMI_OUTPUT = "\n".join(
    [
        "",
        "    Instantaneous power reading:                   290 Watts",
        "    Minimum during sampling period:                104 Watts",
        "    Maximum during sampling period:                782 Watts",
        "    Average power reading over sample period:      261 Watts",
        "    IPMI timestamp:                10/23/2026 10:08:41 AM EDT"
        "    Sampling period:                02589411 Seconds.",
        "    Power reading state is:                   activated",
        "",
    ]
)

### Unit Tests ###


def test_system_samples_reads_instantaneous_watts(monkeypatch):
    """Parse the instantaneous line, not the min/max/average junk."""
    monkeypatch.setattr(
        samplers.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(stdout=DCMI_OUTPUT),
    )

    stream = samplers.system_samples()

    assert next(stream) == {"power_w": 290}
    stream.close()
