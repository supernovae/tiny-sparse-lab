"""Lightweight pipeline contracts; no registry/network, wheels or containers."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import image_plan


class ImagePlanTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in image_plan.INPUTS:
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                name
                if not name.endswith("build-requirements.txt")
                else "uv-build==0.12.17 --hash=sha256:" + "a" * 64
            )
        self.backend = self.root / "container/backends/cuda"
        self.backend.mkdir(parents=True)
        (self.backend / "config.json").write_text(
            json.dumps(
                {
                    "backend": "cuda",
                    "torch_version": "2.14.1+cu132",
                    "runtime_version": "13.2",
                    "platform": "linux/amd64",
                }
            )
        )
        (self.backend / "requirements.lock").write_text(
            "torch==2.14.1+cu132 --hash=sha256:" + "a" * 64
        )

    def test_every_dependency_input_changes_identity(self):
        before = image_plan.identity(self.root, "cuda")
        for name in (*image_plan.INPUTS, "container/backends/cuda/requirements.lock"):
            path = self.root / name
            original = path.read_text()
            path.write_text(original + "\n# changed")
            self.assertNotEqual(before, image_plan.identity(self.root, "cuda"), name)
            path.write_text(original)
        cfg = self.backend / "config.json"
        cfg.write_text(cfg.read_text().replace("13.2", "13.4"))
        self.assertNotEqual(before, image_plan.identity(self.root, "cuda"))

    def test_source_and_application_recipe_reuse_dependencies(self):
        before = image_plan.identity(self.root, "cuda")
        (self.root / "src").mkdir()
        (self.root / "src/app.py").write_text("changed")
        (self.root / "container/application.Dockerfile").write_text("changed")
        self.assertEqual(before, image_plan.identity(self.root, "cuda"))

    def test_invalid_config_and_missing_input_refuse(self):
        with self.assertRaises(ValueError):
            image_plan.identity(self.root, "../cuda")
        (self.backend / "requirements.lock").unlink()
        with self.assertRaises(ValueError):
            image_plan.identity(self.root, "cuda")

    def test_registry_errors_are_not_bootstrap(self):
        for error in ("unauthorized", "denied", "connection reset", "not found"):
            with (
                patch(
                    "image_plan.subprocess.run",
                    return_value=subprocess.CompletedProcess([], 1, "", error),
                ),
                self.assertRaises(RuntimeError),
            ):
                image_plan.registry_digest("example")
        with patch(
            "image_plan.subprocess.run",
            return_value=subprocess.CompletedProcess([], 1, "", "manifest unknown"),
        ):
            self.assertIsNone(image_plan.registry_digest("example"))

    def test_unhashed_or_unpinned_lock_refuses(self):
        for value in (
            "torch>=2",
            "torch==2",
            "--extra-index-url https://untrusted.example",
        ):
            (self.backend / "requirements.lock").write_text(value)
            with self.assertRaises(ValueError):
                image_plan.identity(self.root, "cuda")

    def test_registry_digest_is_exact(self):
        digest = "sha256:" + "a" * 64
        with patch(
            "image_plan.subprocess.run",
            return_value=subprocess.CompletedProcess(
                [], 0, json.dumps({"digest": digest}), ""
            ),
        ):
            self.assertEqual(digest, image_plan.registry_digest("example"))

    def test_existing_dependencies_are_never_rebuilt(self):
        digest = "sha256:" + "b" * 64
        with (
            patch("image_plan.registry_digest", return_value=digest),
            patch("image_plan.run") as run,
            patch("image_plan.subprocess.run"),
            patch("image_plan.Path.write_text"),
        ):
            image_plan.publish(
                self.root, "cuda", "ghcr.io/supernovae/tiny-sparse-lab", "c" * 40
            )
        calls = [call.args[0] for call in run.call_args_list]
        builds = [cmd for cmd in calls if cmd[:3] == ["docker", "buildx", "build"]]
        self.assertEqual(len(builds), 1)
        self.assertIn(
            "DEPENDENCY_IMAGE=ghcr.io/supernovae/tiny-sparse-lab@" + digest, builds[0]
        )
        self.assertEqual(calls[-1][:2], ["docker", "push"])

    def test_failed_smoke_prevents_application_push(self):
        def fake_run(argv, **kwargs):
            if argv[:2] == ["docker", "run"]:
                raise RuntimeError("smoke failed")

        with (
            patch("image_plan.registry_digest", return_value="sha256:" + "a" * 64),
            patch("image_plan.run", side_effect=fake_run) as run,
            patch("image_plan.subprocess.run"),
            self.assertRaises(RuntimeError),
        ):
            image_plan.publish(
                self.root, "cuda", "ghcr.io/supernovae/tiny-sparse-lab", "c" * 40
            )
        self.assertFalse(
            any(c.args[0][:2] == ["docker", "push"] for c in run.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
