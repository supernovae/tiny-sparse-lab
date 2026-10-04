from __future__ import annotations

import sys

import pytest

from sparselab.cli.main import main


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--tokenizer-batch-documents", "0"),
        ("--tokenizer-batch-documents", "257"),
        ("--tokenizer-batch-documents", "1.5"),
        ("--tokenizer-batch-documents", "true"),
        ("--tokenizer-batch-source-bytes", "0"),
        ("--tokenizer-batch-source-bytes", "8388609"),
    ],
)
def test_invalid_batch_limit_fails_before_workspace(tmp_path, monkeypatch, flag, value):
    work = tmp_path / "must-not-exist"
    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(work))
    monkeypatch.setattr(
        sys,
        "argv",
        ["sparselab", "data", "prepare", str(tmp_path / "missing.yaml"), flag, value],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not work.exists()
