"""Human-facing inference policy and provenance controls, using offline doubles."""

import json
import sys
from types import SimpleNamespace

import pytest

from sparselab.cli import main as cli
from sparselab.dashboard.surface_chat import generate_comparison


@pytest.fixture
def loaded():
    return SimpleNamespace(
        model=object(),
        tokenizer=SimpleNamespace(
            encode=lambda text, **kwargs: SimpleNamespace(ids=list(range(len(text))))
        ),
        config=SimpleNamespace(model=SimpleNamespace(max_seq_len=100)),
        device="cpu",
        engine=None,
        identity={"run_id": "fixture", "checkpoint_sha256": "verified-digest"},
    )


@pytest.mark.parametrize("command", ["chat", "generate"])
def test_cli_records_and_forwards_debug_controls(command, loaded, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(cli, "load_run", lambda *args, **kwargs: loaded)

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return args[2] + "answer"

    monkeypatch.setattr(cli, "generate", generate)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sparselab",
            command,
            "fixture",
            "--message" if command == "chat" else "--prompt",
            "hello",
            "--json",
            "--show-prompt",
            "--stop",
            "END",
            "--no-cache",
            "--context-length",
            "80",
            "--max-new-tokens",
            "2",
            "--seed",
            "7",
        ],
    )
    args = cli.build_parser().parse_args(sys.argv[1:])
    args.runtime_authorization = None
    args.handler(args)
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert json.loads(captured.err)["prompt"] == result["prompt"]
    assert result["identity"] == loaded.identity
    assert result["generation"]["context_length"] == 80
    assert result["generation"]["seed"] == 7
    assert result["generation"]["use_cache"] is False
    assert "END" in result["generation"]["stop_sequences"]
    assert calls[0][0][3] == 80
    assert calls[0][1]["use_cache"] is False
    assert result["response"] == "answer"


def test_context_cannot_exceed_native():
    with pytest.raises(ValueError, match="between 1 and 100"):
        cli._inference_context(SimpleNamespace(context_length=101), 100)


def test_comparison_records_policy_and_per_checkpoint_context(loaded, monkeypatch):
    calls = []

    def generate(*args, **kwargs):
        calls.append((args, kwargs))
        return args[2] + "answer", [3, 4]

    monkeypatch.setattr(
        "sparselab.evaluation.generation.generate_with_token_ids", generate
    )
    result = generate_comparison(
        "hello\n",
        [("first", loaded), ("second", loaded)],
        max_new_tokens=2,
        temperature=0.7,
        top_k=3,
        seed=9,
        ordinal=2,
        stop_sequences=["END"],
        use_cache=False,
        context_length=50,
    )
    assert result["prompt"] == "hello\n"
    assert result["prompt_format"] == "raw"
    assert result["policy"]["seed"] == 11
    assert result["policy"]["stop_sequences"] == ["END"]
    assert all(
        card["prompt_tokens"] == 6 and card["context_length"] == 50
        for card in result["cards"]
    )
    assert all(args[3] == 50 and kwargs["use_cache"] is False for args, kwargs in calls)


@pytest.mark.parametrize(
    "options",
    [{"context_length": 101}, {"context_length": 5}, {"stop_sequences": [""]}],
)
def test_comparison_preflights_all_controls_before_generation(
    loaded, monkeypatch, options
):
    def unexpected(*args, **kwargs):
        pytest.fail("generation started before preflight completed")

    monkeypatch.setattr(
        "sparselab.evaluation.generation.generate_with_token_ids", unexpected
    )
    with pytest.raises(ValueError):
        generate_comparison(
            "hello",
            [("a", loaded), ("b", loaded)],
            max_new_tokens=2,
            temperature=0,
            top_k=0,
            seed=0,
            ordinal=0,
            **options,
        )


@pytest.mark.parametrize("finished", [False, True])
def test_surface_reveals_identity_only_after_finish(loaded, monkeypatch, finished):
    from contextlib import nullcontext

    from sparselab.dashboard.surface_chat import render_chat

    loaded.identity["checkpoint_relative_path"] = "checkpoints/step_1"
    card = {
        "label": "A",
        "alias": "private-alias",
        "identity": loaded.identity,
        "response": "answer",
        "prompt_tokens": 5,
        "context_length": 100,
        "token_ids": [1],
    }
    turn = {
        "ordinal": 0,
        "prompt": "hello",
        "policy": {"seed": 0},
        "cards": [card],
        "vote": None,
    }

    class State(dict):
        __getattr__ = dict.__getitem__
        __setattr__ = dict.__setitem__

    output = []
    state = State(
        surface_chat_signature=(("a", "b"), None, 0),
        surface_chat_checkpoints=[("a", loaded), ("b", loaded)],
        surface_chat_turns=[turn],
        surface_chat_finished=finished,
        surface_chat_saved=None,
    )
    fake = SimpleNamespace(session_state=state)
    for name in (
        "title",
        "caption",
        "subheader",
        "code",
        "json",
        "markdown",
        "text",
        "success",
        "write",
    ):
        setattr(fake, name, lambda value, **kwargs: output.append(value))
    fake.selectbox = lambda label, choices, **kwargs: choices[0]
    fake.form = lambda *args, **kwargs: nullcontext()
    fake.text_area = lambda *args, **kwargs: ""
    fake.number_input = lambda *args, **kwargs: kwargs["value"]
    fake.radio = lambda *args, **kwargs: "Greedy"
    fake.text_input = lambda *args, **kwargs: "[]"
    fake.checkbox = lambda *args, **kwargs: True
    fake.form_submit_button = fake.button = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, "streamlit", fake)
    render_chat(SimpleNamespace(cell=["a", "b"], backend=None, seed=0))
    visible = json.dumps(output)
    assert ("verified-digest" in visible) is finished
    assert ("private-alias" in visible) is finished
    assert "hello" in visible
