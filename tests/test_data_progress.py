from __future__ import annotations

import json

from sparselab.data.progress import heartbeat


def test_preprocessing_heartbeat_emits_start_and_finish(capsys) -> None:
    with heartbeat("blocked-stage"):
        pass
    captured = capsys.readouterr()
    events = [json.loads(line) for line in captured.err.splitlines()]
    assert captured.out == ""
    assert [event["event"] for event in events] == ["started", "finished"]
    assert all(event["preprocessing"] == "blocked-stage" for event in events)
