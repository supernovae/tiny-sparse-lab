"""Native monitor sensor and storage guarantees from retired shell launchers.

AMD SMI is mocked; these tests neither initialize models nor access a GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sparselab import operational_monitor as monitor


def _fake_amdsmi(
    monkeypatch: pytest.MonkeyPatch, *, used: int = 100
) -> SimpleNamespace:
    device = object()
    fake = SimpleNamespace(
        AmdSmiMemoryType=SimpleNamespace(VRAM=object()),
        amdsmi_init=Mock(),
        amdsmi_shut_down=Mock(),
        amdsmi_get_processor_handles=Mock(return_value=[device]),
        amdsmi_get_gpu_device_uuid=Mock(return_value="expected-uuid"),
        amdsmi_get_gpu_memory_usage=Mock(return_value=used),
        amdsmi_get_gpu_memory_total=Mock(return_value=1000),
    )
    monkeypatch.setitem(sys.modules, "amdsmi", fake)
    return fake


def test_reader_returns_whole_device_bytes_and_checks_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _fake_amdsmi(monkeypatch)
    assert monitor._read_device_memory_direct("expected-uuid") == 100
    fake.amdsmi_get_gpu_memory_usage.assert_called_once_with(
        fake.amdsmi_get_processor_handles.return_value[0],
        fake.AmdSmiMemoryType.VRAM,
    )
    fake.amdsmi_shut_down.assert_called_once()


@pytest.mark.parametrize(
    "failure",
    [
        "absent",
        "ambiguous",
        "uuid",
        "negative",
        "over_total",
        "bool",
        "float",
        "zero_total",
        "api_error",
    ],
)
def test_reader_fails_closed_on_untrusted_reading(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    fake = _fake_amdsmi(monkeypatch)
    if failure == "absent":
        fake.amdsmi_get_processor_handles.return_value = []
    elif failure == "ambiguous":
        fake.amdsmi_get_processor_handles.return_value = [object(), object()]
    elif failure == "uuid":
        fake.amdsmi_get_gpu_device_uuid.return_value = "other-uuid"
    elif failure == "negative":
        fake.amdsmi_get_gpu_memory_usage.return_value = -1
    elif failure == "over_total":
        fake.amdsmi_get_gpu_memory_usage.return_value = 1001
    elif failure == "bool":
        fake.amdsmi_get_gpu_memory_usage.return_value = True
    elif failure == "float":
        fake.amdsmi_get_gpu_memory_usage.return_value = float("nan")
    elif failure == "zero_total":
        fake.amdsmi_get_gpu_memory_total.return_value = 0
    else:
        fake.amdsmi_get_gpu_memory_usage.side_effect = RuntimeError("sensor lost")
    with pytest.raises(RuntimeError):
        monitor._read_device_memory_direct("expected-uuid")
    fake.amdsmi_shut_down.assert_called_once()


def test_workspace_sample_counts_live_sqlite_and_hardlinks_once(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    data = nested / "data.bin"
    data.write_bytes(b"data" * 1024)
    (nested / "second-link").hardlink_to(data)
    (tmp_path / "db.sqlite-wal").write_bytes(b"W" * 4096)
    (tmp_path / "db.sqlite-shm").write_bytes(b"S" * 1024)
    apparent_bytes, inodes = monitor.sample_workspace_tree(tmp_path)
    assert apparent_bytes == 4096 + 4096 + 1024
    assert inodes == 6  # root, nested, both links, live WAL and SHM
