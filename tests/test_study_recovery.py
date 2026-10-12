"""Model-free publication fault injection; submission never prepares a bundle."""

import json
from types import SimpleNamespace

import pytest
from test_study import _variants, _write_study

from sparselab.experiments import study as subject
from sparselab.workers.controller import Controller
from sparselab.workers.store import ControllerStore


@pytest.fixture
def submission(tmp_path, monkeypatch):
    study = subject.plan_study(_write_study(tmp_path, _variants(), [], seeds=False))
    store = tmp_path / "runs"
    receipt = tmp_path / "receipt.json"
    calls = []

    def fake_submit(controller, requests):
        calls.append(requests)
        entries = []
        for index, request in enumerate(requests):
            config = request["config"].model_dump(mode="json")
            entries.append(
                {
                    "experiment_id": f"experiment-{index}",
                    "attempt_id": f"attempt-{index}",
                    "run_id": f"run-{index}",
                    "spec": {
                        "experiment_id": f"experiment-{index}",
                        "config": config,
                        "config_sha256": subject.config_sha256(config),
                        "matrix": request["matrix"],
                        "declaration_provenance": request.get("declaration_provenance"),
                    },
                }
            )
        controller.store.enqueue_many(entries)
        return [
            SimpleNamespace(
                **{key: entry[key] for key in ("experiment_id", "attempt_id", "run_id")}
            )
            for entry in entries
        ]

    monkeypatch.setattr(Controller, "submit_many", fake_submit)
    return study, store, receipt, calls


def test_injected_publication_failure_retains_native_jobs(submission, monkeypatch):
    study, store, receipt, calls = submission
    writer = subject._write_json_exclusive

    def fail(path, payload):
        if path == receipt:
            path.write_bytes(b'{"partial":')
            raise OSError("injected publication failure")
        writer(path, payload)

    monkeypatch.setattr(subject, "_write_json_exclusive", fail)
    with pytest.raises(OSError, match="study runs were submitted"):
        subject.submit_study(study, receipt, store=store)
    assert len(calls) == 1
    assert len(ControllerStore(store, read_only=True).attempts()) == 3
    assert receipt.read_bytes() == b'{"partial":'
    with pytest.raises(FileExistsError):
        subject.submit_study(study, receipt, store=store)
    assert len(calls) == 1


def recover(submission, output):
    study, store, receipt, _ = submission
    return subject.recover_study_receipt(
        study, subject.study_intent_path(receipt), output, store=store
    )


@pytest.mark.parametrize("original", [None, b'{"partial":', b'{"old":"evidence"}'])
def test_recovery_preserves_original_and_never_dispatches(
    submission, monkeypatch, original
):
    study, store, receipt, calls = submission
    expected = subject.submit_study(study, receipt, store=store)
    if original is None:
        receipt.unlink()
    else:
        receipt.write_bytes(original)

    def forbidden(*args, **kwargs):
        pytest.fail("recovery attempted controller/worker execution")

    monkeypatch.setattr(Controller, "__init__", forbidden)
    monkeypatch.setattr(Controller, "submit_many", forbidden)
    monkeypatch.setattr(Controller, "submit", forbidden)
    monkeypatch.setattr(Controller, "tick", forbidden)
    output = receipt.parent / "recovered.json"
    result = recover(submission, output)
    assert result["runs"] == expected["runs"]
    parsed, _ = subject._read_receipt(output)
    assert parsed == result
    assert parsed["cards"] == expected["cards"]
    assert len(calls) == 1
    assert (receipt.read_bytes() if receipt.exists() else None) == original
    assert result["recovery"]["original_receipt_sha256"] == (
        subject.hashlib.sha256(original).hexdigest() if original is not None else None
    )


@pytest.mark.parametrize("target", ["receipt", "intent"])
@pytest.mark.parametrize("content", [b"", b'{"partial":', b'{"complete":true}'])
def test_preexisting_evidence_prevents_enqueue(submission, target, content):
    study, store, receipt, calls = submission
    path = receipt if target == "receipt" else subject.study_intent_path(receipt)
    path.write_bytes(content)
    with pytest.raises(FileExistsError):
        subject.submit_study(study, receipt, store=store)
    assert not calls
    assert path.read_bytes() == content


@pytest.mark.parametrize("when", ["before", "after"])
def test_interrupted_enqueue_is_recoverable_only_after_transaction(
    submission, monkeypatch, when
):
    study, store, receipt, _calls = submission
    submit = Controller.submit_many

    def interrupt(controller, requests):
        if when == "after":
            submit(controller, requests)
        raise KeyboardInterrupt("injected enqueue interruption")

    monkeypatch.setattr(Controller, "submit_many", interrupt)
    with pytest.raises(KeyboardInterrupt):
        subject.submit_study(study, receipt, store=store)
    assert subject.study_intent_path(receipt).is_file()
    assert not receipt.exists()
    output = receipt.parent / "recovered.json"
    if when == "after":
        assert len(recover(submission, output)["runs"]) == 3
    else:
        with pytest.raises(ValueError):
            recover(submission, output)
        assert not output.exists()


@pytest.mark.parametrize("failure", [OSError, RuntimeError, KeyboardInterrupt])
def test_publication_exception_keeps_intent_and_jobs(submission, monkeypatch, failure):
    study, store, receipt, calls = submission
    writer = subject._write_json_exclusive

    def fail(path, payload):
        if path == receipt:
            raise failure("injected")
        writer(path, payload)

    monkeypatch.setattr(subject, "_write_json_exclusive", fail)
    with pytest.raises(KeyboardInterrupt if failure is KeyboardInterrupt else OSError):
        subject.submit_study(study, receipt, store=store)
    output = receipt.parent / "recovered.json"
    assert len(recover(submission, output)["runs"]) == 3
    assert len(calls) == 1


@pytest.mark.parametrize(
    "field",
    [
        "study_sha256",
        "matrix_sha256",
        "cards_sha256",
        "intent_sha256",
        "config_sha256",
        "submission_id",
        "index",
        "coordinate",
        "legacy",
        "config",
        "matrix",
        "experiment_id",
        "missing",
        "duplicate",
    ],
)
def test_native_identity_tampering_fails_closed(submission, field):
    study, store, receipt, _ = submission
    subject.submit_study(study, receipt, store=store)
    queue = ControllerStore(store)
    rows = queue.attempts()
    spec = rows[0]["spec"]
    binding = spec["declaration_provenance"]["study_submission"]
    if field.endswith("sha256"):
        binding[field] = "f" * 64
    elif field == "submission_id":
        binding[field] = "f" * 32
    elif field == "index":
        binding[field] = 1
    elif field == "coordinate":
        binding[field] = {"architecture": "other"}
    elif field == "legacy":
        spec.pop("declaration_provenance")
    elif field == "config":
        spec["config"]["seed"] += 1
    elif field == "matrix":
        spec["matrix"]["coordinate"] = {"architecture": "other"}
    elif field == "experiment_id":
        spec[field] = "other"
    with queue.transaction() as con:
        if field == "missing":
            con.execute(
                "DELETE FROM attempts WHERE attempt_id=?", (rows[0]["attempt_id"],)
            )
        elif field == "duplicate":
            con.execute(
                "INSERT INTO attempts(attempt_id,experiment_id,run_id,status) VALUES(?,?,?,'QUEUED')",
                ("duplicate", rows[0]["experiment_id"], "duplicate-run"),
            )
        else:
            con.execute(
                "UPDATE experiments SET specification_json=? WHERE experiment_id=?",
                (json.dumps(spec), rows[0]["experiment_id"]),
            )
    output = receipt.parent / "recovered.json"
    with pytest.raises(ValueError):
        recover(submission, output)
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate-key", "card", "store", "submission", "version"]
)
def test_intent_tampering_fails_closed(submission, mutation):
    study, store, receipt, _ = submission
    subject.submit_study(study, receipt, store=store)
    path = subject.study_intent_path(receipt)
    intent = json.loads(path.read_bytes())
    if mutation == "missing":
        path.unlink()
    elif mutation == "duplicate-key":
        path.write_text('{"format":"a","format":"b"}')
    else:
        if mutation == "card":
            intent["receipt"]["cards"][0]["digest"] = "f" * 64
        elif mutation == "store":
            intent["store"] = str(store.parent)
        elif mutation == "submission":
            intent["submission_id"] = "f" * 32
        else:
            intent["schema_version"] = True
        path.write_text(json.dumps(intent))
    with pytest.raises((ValueError, FileNotFoundError)):
        recover(submission, receipt.parent / "recovered.json")


@pytest.mark.parametrize("boundary", ["short", "flush", "fsync", "directory", "link"])
def test_atomic_writer_failure_boundaries(tmp_path, monkeypatch, boundary):
    path = tmp_path / "receipt.json"
    payload = {"complete": True}
    original_fdopen = subject.os.fdopen
    if boundary in {"short", "flush"}:

        class FaultyHandle:
            def __init__(self, descriptor, mode):
                self.handle = original_fdopen(descriptor, mode)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.handle.close()

            def write(self, data):
                if boundary == "short":
                    return self.handle.write(data[:2])
                return self.handle.write(data)

            def flush(self):
                raise RuntimeError("flush failure")

            def fileno(self):
                return self.handle.fileno()

        monkeypatch.setattr(subject.os, "fdopen", FaultyHandle)
    elif boundary == "fsync":
        monkeypatch.setattr(
            subject.os, "fsync", lambda _: (_ for _ in ()).throw(OSError("fsync"))
        )
    elif boundary == "directory":
        monkeypatch.setattr(
            subject,
            "_sync_directory",
            lambda _: (_ for _ in ()).throw(OSError("directory fsync")),
        )
    else:

        def concurrent(source, destination):
            destination.write_bytes(b"concurrent evidence")
            raise FileExistsError("concurrent publisher")

        monkeypatch.setattr(subject.os, "link", concurrent)
    with pytest.raises((OSError, RuntimeError)):
        subject._write_json_exclusive(path, payload)
    if boundary == "directory":
        assert json.loads(path.read_bytes()) == payload
    elif boundary == "link":
        assert path.read_bytes() == b"concurrent evidence"
    else:
        assert not path.exists()
    assert list(tmp_path.iterdir()) == ([path] if path.exists() else [])


def test_intent_fsync_uncertainty_never_enqueues(submission, monkeypatch):
    study, store, receipt, calls = submission
    monkeypatch.setattr(
        subject,
        "_sync_directory",
        lambda _: (_ for _ in ()).throw(OSError("uncertain")),
    )
    with pytest.raises(OSError):
        subject.submit_study(study, receipt, store=store)
    assert not calls
    assert json.loads(subject.study_intent_path(receipt).read_bytes())["receipt"][
        "runs"
    ]


def test_concurrent_recovery_publishers_have_one_winner(submission):
    from concurrent.futures import ThreadPoolExecutor

    study, store, receipt, calls = submission
    subject.submit_study(study, receipt, store=store)
    output = receipt.parent / "recovered.json"

    def publish():
        try:
            return recover(submission, output)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert (
        json.loads(output.read_bytes())["runs"]
        == next(result for result in results if result)["runs"]
    )
    assert len(calls) == 1


def test_recovery_output_collision_preserves_bytes(submission):
    study, store, receipt, _calls = submission
    subject.submit_study(study, receipt, store=store)
    original = receipt.read_bytes()
    with pytest.raises(FileExistsError):
        recover(submission, receipt)
    assert receipt.read_bytes() == original


def test_collector_accepts_recovered_receipt_without_model_work(
    submission, monkeypatch
):
    study, store, receipt, _ = submission
    subject.submit_study(study, receipt, store=store)
    output = receipt.parent / "recovered.json"
    result = recover(submission, output)

    def unavailable(*args, **kwargs):
        raise FileNotFoundError("no checkpoint; model work forbidden")

    monkeypatch.setattr(subject, "_evaluate_trial", unavailable)
    report, path = subject.collect_study(study, output, runs_dir=store)
    assert [run["run_id"] for run in report["runs"]] == [
        run["run_id"] for run in result["runs"]
    ]
    assert all(run["status"] == "unavailable" for run in report["runs"])
    assert path.is_file()


def test_duplicate_persisted_json_keys_rejected(submission):
    study, store, receipt, _ = submission
    subject.submit_study(study, receipt, store=store)
    queue = ControllerStore(store)
    row = queue.attempts()[0]
    encoded = json.dumps(row["spec"])
    encoded = '{"experiment_id":"hidden",' + encoded[1:]
    with queue.transaction() as connection:
        connection.execute(
            "UPDATE experiments SET specification_json=? WHERE experiment_id=?",
            (encoded, row["experiment_id"]),
        )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        recover(submission, receipt.parent / "recovered.json")


def test_native_spec_validates_namespaced_binding(submission):
    from pydantic import ValidationError

    from sparselab.workers.models import ExperimentSpec

    study, store, receipt, _ = submission
    subject.submit_study(study, receipt, store=store)
    binding = ControllerStore(store, read_only=True).attempts()[0]["spec"][
        "declaration_provenance"
    ]
    assert ExperimentSpec.study_binding(binding) == binding
    binding["study_submission"]["index"] = True
    with pytest.raises(ValidationError):
        ExperimentSpec.study_binding(binding)


def test_concurrent_submitters_enqueue_once(submission):
    from concurrent.futures import ThreadPoolExecutor

    study, store, receipt, calls = submission

    def publish():
        try:
            return subject.submit_study(study, receipt, store=store)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: publish(), range(2)))
    assert sum(result is not None for result in results) == 1
    assert len(calls) == 1
    assert len(ControllerStore(store, read_only=True).attempts()) == 3


def test_recover_cli_reads_only_bound_native_state(submission, monkeypatch, capsys):
    from sparselab.cli.main import build_parser

    study, store, receipt, calls = submission
    subject.submit_study(study, receipt, store=store)
    queue = ControllerStore(store, read_only=True)
    before = queue.path.read_bytes()
    output = receipt.parent / "recovered.json"
    args = build_parser(work_dir=receipt.parent).parse_args(
        [
            "study",
            "recover",
            str(study.path),
            "--intent",
            str(subject.study_intent_path(receipt)),
            "--store",
            str(store),
            "--output",
            str(output),
        ]
    )

    def forbidden(*args, **kwargs):
        pytest.fail("recovery invoked execution")

    for name in (
        "__init__",
        "submit",
        "submit_many",
        "resume",
        "tick",
        "run",
        "_launch",
        "_prepare_entry",
        "_rpc_result",
    ):
        monkeypatch.setattr(Controller, name, forbidden)
    args.handler(args)
    assert (
        json.loads(capsys.readouterr().out)["runs"]
        == json.loads(receipt.read_bytes())["runs"]
    )
    assert queue.path.read_bytes() == before
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["receipt", "recovery"])
def test_directory_fsync_failure_preserves_surviving_complete_receipt(
    submission, monkeypatch, kind
):
    study, store, receipt, calls = submission
    sync = subject._sync_directory
    if kind == "recovery":
        subject.submit_study(study, receipt, store=store)
    count = 0

    def fail_directory(path):
        nonlocal count
        count += 1
        if kind == "recovery" or count == 2:
            raise OSError("directory durability uncertain")
        sync(path)

    monkeypatch.setattr(subject, "_sync_directory", fail_directory)
    output = receipt if kind == "receipt" else receipt.parent / "recovered.json"
    with pytest.raises(OSError):
        if kind == "receipt":
            subject.submit_study(study, receipt, store=store)
        else:
            recover(submission, output)
    assert len(json.loads(output.read_bytes())["runs"]) == 3
    assert len(calls) == 1
    if kind == "recovery":
        with pytest.raises(FileExistsError):
            recover(submission, output)


@pytest.mark.parametrize("key", ["attempt_id", "run_id"])
def test_unsafe_native_ids_rejected(submission, key):
    study, store, receipt, _ = submission
    subject.submit_study(study, receipt, store=store)
    queue = ControllerStore(store)
    row = queue.attempts()[0]
    with queue.transaction() as connection:
        connection.execute(
            f"UPDATE attempts SET {key}=? WHERE attempt_id=?",
            ("../unsafe", row["attempt_id"]),
        )
    with pytest.raises(ValueError):
        recover(submission, receipt.parent / "recovered.json")


@pytest.mark.parametrize("mask", [0o022, 0o077])
def test_atomic_publication_retains_existing_creation_permissions(tmp_path, mask):
    import os
    import stat

    previous = os.umask(mask)
    try:
        reference = tmp_path / "reference"
        with reference.open("xb") as handle:
            handle.write(b"reference")
        output = tmp_path / "receipt.json"
        subject._write_json_exclusive(output, {"complete": True})
    finally:
        os.umask(previous)
    assert stat.S_IMODE(output.stat().st_mode) == stat.S_IMODE(reference.stat().st_mode)
