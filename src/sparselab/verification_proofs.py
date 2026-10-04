"""Host-local authenticated operational receipts, never artifact identity authority.

Trust assumes immutable artifacts managed by the service UID. This does not
protect against a compromised same-UID owner. Every miss requires cold checking.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import stat
from pathlib import Path
from typing import Literal

from sparselab.training.manifest import canonical_json, source_identity

VERIFIER_VERSION = 1
VerificationMode = Literal["cold", "verified_reuse"]


def validate_mode(mode: VerificationMode) -> None:
    if mode not in {"cold", "verified_reuse"}:
        raise ValueError(f"unknown verification mode: {mode}")


def _json_native(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (tuple, list)):
        return [_json_native(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_native(item) for key, item in value.items()}
    return value


def _fingerprint(path: Path) -> list[int]:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) and not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"nonregular verification path: {path}")
    return [
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    ]


def _trusted(root: Path, path: Path, *, private: bool = False) -> bool:
    """Check selected owned root and children, not system-owned ancestors."""
    try:
        root, path = root.absolute(), path.absolute()
        if root == Path(root.anchor) or not path.is_relative_to(root):
            return False
        # Even a root reached through a link is not a registered trust domain.
        if any(parent.is_symlink() for parent in (root, *root.parents)):
            return False
        entries = [root]
        relative = path.relative_to(root)
        current = root
        for part in relative.parts:
            current = current / part
            entries.append(current)
        for entry in entries:
            info = entry.lstat()
            if info.st_uid != os.getuid() or info.st_mode & 0o022:
                return False
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                return False
            if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
                return False
        if private:
            info = path.lstat()
            expected = 0o700 if stat.S_ISDIR(info.st_mode) else 0o600
            if stat.S_IMODE(info.st_mode) != expected:
                return False
        return True
    except OSError, ValueError:
        return False


def file_binding(
    path: Path,
    sha256: str,
    *,
    kind: str = "file",
    version: int = 1,
    identifier: str | None = None,
    closure: object = None,
) -> dict[str, object]:
    path = path.absolute()
    return {
        "kind": kind,
        "version": version,
        "identifier": identifier or path.name,
        "sha256": sha256,
        "path": str(path),
        "members": [[".", *_fingerprint(path)]],
        "manifest_inventory_sha256": hashlib.sha256(
            canonical_json(closure)
        ).hexdigest(),
        "upstream_identities": _json_native(closure),
        "dependencies": [],
        "success": True,
    }


def artifact_binding(
    key: tuple[object, ...], identity: dict[str, object]
) -> dict[str, object]:
    manifests = json.loads(key[8])
    return {
        **identity,
        "state": key[7],
        "members": _json_native(key[5]),
        "dependencies": _json_native(key[6]),
        "manifest_inventory_sha256": hashlib.sha256(
            canonical_json(manifests)
        ).hexdigest(),
        "upstream_identities": manifests["upstream"],
        "success": True,
    }


def _current(binding: dict[str, object]) -> bool:
    try:
        path = Path(str(binding["path"]))
        members = binding["members"]
        if not isinstance(members, list):
            return False
        actual = [
            [p.relative_to(path).as_posix() if p != path else ".", *_fingerprint(p)]
            for p in ([path] if path.is_file() else [path, *sorted(path.rglob("*"))])
        ]
        if actual != members:
            return False
        for dependency, fingerprints in binding["dependencies"]:
            root = Path(dependency)
            actual = [
                [p.relative_to(root).as_posix() if p != root else ".", *_fingerprint(p)]
                for p in (
                    [root] if root.is_file() else [root, *sorted(root.rglob("*"))]
                )
            ]
            if actual != fingerprints:
                return False
        return True
    except OSError, KeyError, TypeError, ValueError:
        return False


class ProofStore:
    """One signed receipt index under an explicitly selected task-owned work root."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).absolute()
        self.directory = self.root / "cache" / "verification-v1"
        configured = os.environ.get("XDG_CONFIG_HOME", "")
        self.config_root = (
            Path(configured)
            if configured and Path(configured).is_absolute()
            else Path.home() / ".config"
        )
        self.key_path = self.config_root / "sparselab" / "verification-key-v1"
        self.source_sha256 = source_identity()["sha256"]
        self.hits = self.misses = self.recorded = 0

    def trusted(self, path: Path) -> bool:
        return _trusted(self.root, path)

    def _safe_binding(self, binding: dict[str, object]) -> bool:
        try:
            paths = [Path(str(binding["path"]))]
            root = paths[0]
            paths.extend(root / str(row[0]) for row in binding["members"])
            for dependency, fingerprints in binding["dependencies"]:
                paths.append(Path(dependency))
                paths.extend(Path(dependency) / str(row[0]) for row in fingerprints)
            return all(self.trusted(p) for p in paths) and _current(binding)
        except KeyError, TypeError, ValueError:
            return False

    def _payload(self, binding: dict[str, object]) -> dict[str, object]:
        return {
            "verification_schema": VERIFIER_VERSION,
            "implementation_sha256": self.source_sha256,
            "binding": binding,
        }

    def _receipt(self, binding: dict[str, object]) -> Path:
        locator = {
            key: binding.get(key)
            for key in (
                "kind",
                "version",
                "identifier",
                "sha256",
                "path",
                "manifest_inventory_sha256",
            )
        }
        return self.directory / (
            hashlib.sha256(canonical_json(locator)).hexdigest() + ".json"
        )

    def _key(self, *, create: bool) -> bytes | None:
        if create:
            if any(
                parent.is_symlink()
                for parent in (self.config_root, *self.config_root.parents)
            ):
                return None
            self.config_root.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not _trusted(self.config_root, self.config_root):
                return None
            self.key_path.parent.mkdir(mode=0o700, exist_ok=True)
            if not _trusted(self.config_root, self.key_path.parent, private=True):
                return None
            if not self.key_path.exists():
                try:
                    fd = os.open(
                        self.key_path,
                        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                        0o600,
                    )
                except FileExistsError:
                    pass
                else:
                    with os.fdopen(fd, "wb") as stream:
                        stream.write(secrets.token_bytes(32))
                        stream.flush()
                        os.fsync(stream.fileno())
                    _sync(self.key_path.parent)
        if not _trusted(self.config_root, self.key_path, private=True):
            return None
        with self.key_path.open("rb") as stream:
            key = stream.read(33)
        return key if len(key) == 32 else None

    def lookup(self, binding: dict[str, object]) -> bool:
        hit = False
        try:
            receipt = self._receipt(binding)
            if (
                self._safe_binding(binding)
                and _trusted(self.root, self.directory, private=True)
                and _trusted(self.root, receipt, private=True)
            ):
                key = self._key(create=False)
                if key is not None and receipt.stat().st_size <= 16 * 1024 * 1024:
                    raw = json.loads(receipt.read_bytes())
                    payload = self._payload(binding)
                    signature = hmac.new(
                        key, canonical_json(payload), hashlib.sha256
                    ).hexdigest()
                    hit = (
                        isinstance(raw, dict)
                        and raw.get("proof") == payload
                        and isinstance(raw.get("hmac"), str)
                        and hmac.compare_digest(raw["hmac"], signature)
                        and self._safe_binding(binding)
                    )
        except OSError, ValueError, TypeError, KeyError:
            hit = False
        if hit:
            self.hits += 1
        else:
            self.misses += 1
        return hit

    def record(self, binding: dict[str, object], evidence: object) -> None:
        """Only a successful, exact cold verifier's live seal can publish a proof."""
        from sparselab.data.verification import _owned_file
        from sparselab.experiments.artifacts import (
            _MEMO_SEAL,
            _MINTED_ARTIFACTS,
            _VerifiedArtifact,
        )

        owned = False
        if _owned_file(evidence):
            owned = (
                evidence.cold_verified
                and binding.get("path") == str(evidence.path)
                and binding.get("sha256") == evidence.sha256
                and binding.get("members") == [[".", *_fingerprint(evidence.path)]]
            )
        elif isinstance(evidence, _VerifiedArtifact):
            owned = (
                evidence in _MINTED_ARTIFACTS
                and evidence._seal is _MEMO_SEAL
                and evidence.issuer_pid == os.getpid()
                and evidence.cold_verified
                and artifact_binding(evidence.key, dict(evidence.identity)) == binding
            )
        if not owned or binding.get("success") is not True:
            raise TypeError(
                "verification proof requires exact process-owned cold evidence"
            )
        if not self._safe_binding(binding):
            return
        try:
            key = self._key(create=True)
            if key is None:
                return
            self.directory.parent.mkdir(mode=0o700, exist_ok=True)
            if not _trusted(self.root, self.directory.parent):
                return
            self.directory.mkdir(mode=0o700, exist_ok=True)
            if not _trusted(self.root, self.directory, private=True):
                return
            payload = self._payload(binding)
            receipt = self._receipt(binding)
            temporary = self.directory / (".proof-" + secrets.token_hex(16))
            try:
                with temporary.open("xb") as stream:
                    os.chmod(temporary, 0o600)
                    stream.write(
                        canonical_json(
                            {
                                "proof": payload,
                                "hmac": hmac.new(
                                    key, canonical_json(payload), hashlib.sha256
                                ).hexdigest(),
                            }
                        )
                        + b"\n"
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                if not self._safe_binding(binding):
                    raise ValueError("artifact changed during proof publication")
                os.replace(temporary, receipt)
                _sync(self.directory)
                self.recorded += 1
            finally:
                temporary.unlink(missing_ok=True)
        except OSError:
            # Operational cache unavailable: independently cold-verified result stands.
            return


def _sync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def verification_options(root: Path, *, cold: bool = False) -> dict[str, object]:
    if cold or not _trusted(Path(root), Path(root)):
        return {"proof_store": None, "verification_mode": "cold"}
    return {"proof_store": ProofStore(root), "verification_mode": "verified_reuse"}
