"""Hardware observations are passive and never claim an executable backend."""

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from sparselab import runtime_env_hardware as hardware

ROCMINFO = """\
Agent 1
  Name: AMD Ryzen CPU
  Marketing Name: AMD Ryzen 9 7950X
  Device Type: CPU
Agent 2
  Name: gfx1100
  Marketing Name: AMD Radeon RX 7900 XTX
  Device Type: GPU
Agent 3
  Name: gfx1030
  Marketing Name: AMD Radeon RX 6800
  Device Type: GPU
"""


def _responses(monkeypatch, responses):
    def run(command, *, timeout, output_limit):
        assert timeout == 5
        assert output_limit == hardware._OUTPUT_LIMIT
        response = responses[command[0]]
        if isinstance(response, Exception):
            raise response
        output, error, status = response
        return SimpleNamespace(returncode=status, stdout=output, stderr=error)

    monkeypatch.setattr(hardware, "run_bounded", run)


def test_linux_observes_gpu_agents_without_cpu_or_runtime_claim(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    _responses(
        monkeypatch,
        {
            "rocminfo": (ROCMINFO.encode(), b"", 0),
            "nvidia-smi": (b"NVIDIA RTX 6000 Ada Generation\n", b"", 0),
        },
    )
    assert hardware.hardware_inventory() == [
        {
            "backend": "rocm",
            "device_name": "AMD Radeon RX 7900 XTX",
            "observation_source": "rocminfo",
            "status": "OBSERVED",
        },
        {
            "backend": "rocm",
            "device_name": "AMD Radeon RX 6800",
            "observation_source": "rocminfo",
            "status": "OBSERVED",
        },
        {
            "backend": "cuda",
            "device_name": "NVIDIA RTX 6000 Ada Generation",
            "observation_source": "nvidia-smi",
            "status": "OBSERVED",
        },
    ]


@pytest.mark.parametrize(
    "response",
    [
        FileNotFoundError("no tool"),
        subprocess.TimeoutExpired("rocminfo", 5),
        (b"invalid", b"driver error", 1),
        (b"x" * (hardware._OUTPUT_LIMIT + 1), b"", 0),
        (b"", b"x" * (hardware._OUTPUT_LIMIT + 1), 0),
    ],
)
def test_failed_or_oversized_host_tools_are_unknown(monkeypatch, response):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    _responses(monkeypatch, {"rocminfo": response, "nvidia-smi": response})
    assert hardware.hardware_inventory() == [
        {
            "backend": backend,
            "device_name": None,
            "observation_source": source,
            "status": "UNKNOWN",
        }
        for backend, source in (("rocm", "rocminfo"), ("cuda", "nvidia-smi"))
    ]


def test_rocm_cpu_without_gpu_is_unknown(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Linux")
    _responses(
        monkeypatch,
        {
            "rocminfo": (ROCMINFO.split("Agent 2")[0].encode(), b"", 0),
            "nvidia-smi": (b"", b"", 0),
        },
    )
    assert all(item["status"] == "UNKNOWN" for item in hardware.hardware_inventory())


def test_darwin_reports_metal_only_when_supported(monkeypatch):
    monkeypatch.setattr(hardware.platform, "system", lambda: "Darwin")
    displays = {
        "SPDisplaysDataType": [
            {
                "sppci_model": "Apple M4 Max",
                "spdisplays_metal": "Supported, Metal 3",
            },
            {"sppci_model": "Legacy GPU", "spdisplays_metal": "Unsupported"},
        ]
    }
    _responses(
        monkeypatch,
        {"system_profiler": (json.dumps(displays).encode(), b"", 0)},
    )
    assert hardware.hardware_inventory() == [
        {
            "backend": "metal",
            "device_name": "Apple M4 Max",
            "observation_source": "system_profiler",
            "status": "OBSERVED",
        }
    ]
