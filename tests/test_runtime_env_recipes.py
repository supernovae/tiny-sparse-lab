from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

from sparselab import runtime_env_recipes as recipes


@pytest.fixture(autouse=True)
def linux_x86_64_host(monkeypatch):
    """Exercise the Linux recipe independently of the test runner's host."""
    monkeypatch.setattr(recipes.platform, "system", lambda: "Linux")
    monkeypatch.setattr(recipes.platform, "machine", lambda: "x86_64")


def _capacity(
    *, bytes_available=recipes._MIN_BYTES, inodes_available=recipes._MIN_INODES
):
    return SimpleNamespace(
        f_bavail=bytes_available // 4096,
        f_frsize=4096,
        f_favail=inodes_available,
        f_flag=0,
    )


def test_recipe_matches_pinned_vendor_input():
    recipe = recipes.get_recipe("rocm-gfx1100-v1")
    assert recipe.id == "rocm-gfx1100-v1"
    assert recipe.version == 1
    assert recipe.python_version == (3, 14)
    assert recipe.backend == "rocm"
    assert recipe.requirements == {
        "torch_hip": True,
        "bf16": True,
        "device_name_regex": r"Radeon.*7900 XTX",
    }
    assert recipe.requirements_file.is_file()
    lines = [
        line.strip()
        for line in recipe.requirements_file.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert (
        tuple(
            line.split(maxsplit=1)[1]
            for line in lines
            if line.startswith("--extra-index-url ")
        )
        == recipe.indexes
    )
    pins = {
        line.split("==", 1)[0]: line.split("==", 1)[1].split(";", 1)[0].strip()
        for line in lines
        if "==" in line
    }
    assert pins == {
        "triton": "3.8.0+git4cff872c.rocm10.0.0",
        "amd-torch-device-gfx1100": "2.13.0+rocm10.0.0",
        "amd-torch-device-gfx110x": "2.13.0+rocm10.0.0",
        "torch[device-gfx1100]": "2.13.0+rocm10.0.0",
        "rocm[libraries]": "10.0.0",
    }
    assert all("sys_platform == 'linux'" in line for line in lines if "==" in line)
    with pytest.raises(FrozenInstanceError):
        recipe.version = 2
    with pytest.raises(TypeError):
        recipe.requirements["torch_hip"] = False
    with pytest.raises(ValueError, match="unknown runtime recipe"):
        recipes.get_recipe("rocm-gfx1100-v2")


@pytest.mark.parametrize("filesystem", ["ext4", "overlay"])
def test_preflight_private_root_is_passive(tmp_path, monkeypatch, filesystem):
    root = tmp_path / "private" / "runtimes"
    monkeypatch.setattr(recipes, "_filesystem_for", lambda path: (filesystem, tmp_path))
    monkeypatch.setattr(recipes.os, "statvfs", lambda path: _capacity())
    result = recipes.preflight_runtime_root(root)
    assert result == {
        "root": str(root),
        "existing_ancestor": str(tmp_path),
        "filesystem": filesystem,
        "mount_point": str(tmp_path),
        "available_bytes": recipes._MIN_BYTES,
        "available_inodes": recipes._MIN_INODES,
    }
    assert not root.parent.exists()


@pytest.mark.parametrize("kind", ["checkout", "science"])
def test_preflight_rejects_unsafe_roots_without_creating(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "science"))
    root = (
        Path(__file__).resolve().parents[1] / "runtime-unsafe"
        if kind == "checkout"
        else tmp_path / "science"
    )
    with pytest.raises(ValueError, match="checkout|scientific work root"):
        recipes.preflight_runtime_root(root)
    assert not root.exists()


@pytest.mark.parametrize(
    "filesystem", ["nfs", "cifs", "9p", "drvfs", "fuse", "tmpfs", "unknown"]
)
def test_preflight_rejects_nonlocal_or_unknown_mounts(
    tmp_path, monkeypatch, filesystem
):
    monkeypatch.setattr(recipes, "_filesystem_for", lambda path: (filesystem, tmp_path))
    with pytest.raises(ValueError, match="local filesystem"):
        recipes.preflight_runtime_root(tmp_path / "private")
    assert not (tmp_path / "private").exists()


@pytest.mark.parametrize("filesystem", ["ext4", "overlay"])
def test_preflight_rejects_readonly_filesystem(tmp_path, monkeypatch, filesystem):
    monkeypatch.setattr(recipes, "_filesystem_for", lambda path: (filesystem, tmp_path))
    capacity = _capacity()
    capacity.f_flag = recipes.os.ST_RDONLY
    monkeypatch.setattr(recipes.os, "statvfs", lambda path: capacity)
    with pytest.raises(ValueError, match="read-only"):
        recipes.preflight_runtime_root(tmp_path / "private")
    assert not (tmp_path / "private").exists()


@pytest.mark.parametrize(
    "bytes_available,inodes_available,reason",
    [
        (recipes._MIN_BYTES - 4096, recipes._MIN_INODES, "available bytes"),
        (recipes._MIN_BYTES, recipes._MIN_INODES - 1, "available inodes"),
    ],
)
def test_preflight_rejects_low_capacity(
    tmp_path, monkeypatch, bytes_available, inodes_available, reason
):
    monkeypatch.setattr(recipes, "_filesystem_for", lambda path: ("xfs", tmp_path))
    monkeypatch.setattr(
        recipes.os,
        "statvfs",
        lambda path: _capacity(
            bytes_available=bytes_available, inodes_available=inodes_available
        ),
    )
    with pytest.raises(ValueError, match=reason):
        recipes.preflight_runtime_root(tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_mountinfo_decodes_escapes_and_uses_deepest_mount(tmp_path, monkeypatch):
    mount = tmp_path / "with space"
    mount.mkdir()
    nested = mount / "run"
    nested.mkdir()
    escaped = str(mount).replace(" ", "\\040")
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"1 0 1:1 / / rw - ext4 /dev/sda rw\n"
        f"2 1 1:2 / {escaped} rw - nfs server rw\n"
        f"3 2 1:3 / {escaped}/run rw - btrfs /dev/sdb rw\n"
    )
    original_open = Path.open

    def open_mountinfo(path, *args, **kwargs):
        if path == Path("/proc/self/mountinfo"):
            return original_open(mountinfo, *args, **kwargs)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_mountinfo)
    assert recipes._filesystem_for(nested) == ("btrfs", nested)
    assert recipes._filesystem_for(mount) == ("nfs", mount)


def test_preflight_rejects_missing_mount_information(tmp_path, monkeypatch):
    original_open = Path.open

    def unavailable_mountinfo(path, *args, **kwargs):
        if path == Path("/proc/self/mountinfo"):
            raise FileNotFoundError("mountinfo unavailable")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", unavailable_mountinfo)
    with pytest.raises(ValueError, match="cannot inspect local filesystem mounts"):
        recipes.preflight_runtime_root(tmp_path / "private")
    assert not (tmp_path / "private").exists()


def test_preflight_requires_existing_writable_directory_linux(tmp_path, monkeypatch):
    file = tmp_path / "file"
    file.write_text("x")
    with pytest.raises(ValueError, match="not a directory"):
        recipes.preflight_runtime_root(file / "child")
    monkeypatch.setattr(recipes.os, "access", lambda *args, **kwargs: False)
    with pytest.raises(ValueError, match="not writable"):
        recipes.preflight_runtime_root(tmp_path / "private")


@pytest.mark.parametrize(
    "system,machine", [("Darwin", "arm64"), ("Darwin", "x86_64"), ("Linux", "aarch64")]
)
def test_preflight_rejects_unsupported_hosts(tmp_path, monkeypatch, system, machine):
    monkeypatch.setattr(recipes.platform, "system", lambda: system)
    monkeypatch.setattr(recipes.platform, "machine", lambda: machine)
    with pytest.raises(ValueError, match="Linux x86_64"):
        recipes.preflight_runtime_root(tmp_path / "private")
    assert not (tmp_path / "private").exists()
