"""Read-only catalog and static-report views for the dashboard.

This module deliberately consumes only the packaged catalog and validated report
bundles.  It neither opens run storage nor renders bundle-provided HTML/Markdown.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from sparselab.experiments.reporting import load_report_bundle
from sparselab.research.catalog import list_lessons, list_research

_BUNDLE_NAME = re.compile(r"^[0-9a-f]{64}$")
_MAX_BUNDLES = 200


def _plain(value: object) -> object:
    """Convert frozen Pydantic records without relying on a Pydantic version."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value


def _mapping(value: object) -> dict[str, object]:
    value = _plain(value)
    return value if isinstance(value, dict) else {}


def _rows(value: object) -> list[dict[str, object]]:
    value = _plain(value)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, (list, tuple)):
        return []
    return [_mapping(item) for item in value if _mapping(item)]


def _value(record: object, name: str, default: object = "") -> object:
    return _mapping(record).get(name, default)


def _text_list(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return "; ".join(str(item) for item in value)


def _lesson_commands(identifier: str, scale: str, data: str) -> list[str]:
    root = f"experiments/learn-{identifier}"
    return [
        f"sparselab learn scaffold {identifier} --scale {scale} --data {data} --backend cpu --output {root}",
        f"sparselab tokenizer train {root}/tokenizer.yaml",
        f"sparselab data prepare {root}/model.yaml",
        f"sparselab inspect {root}/model.yaml",
        f'sparselab learn probe {root}/model.yaml --prompt "Once upon a time, a small bird found a key."',
        f"sparselab train {root}/model.yaml --run-id learn-{identifier} --stop-after-step 2",
    ]


def learn_page() -> None:
    """Render packaged lessons without requiring a run database."""
    st.header("Learn")
    st.write(
        "Choose one mechanism, inspect its shapes, then copy the standalone commands. No campaign or run store is required to browse these lessons."
    )
    controls = st.columns(2)
    scale = controls[0].selectbox(
        "Scale", ("smoke", "nano", "micro", "tiny"), key="lesson_scale"
    )
    data = controls[1].selectbox("Data", ("offline", "tinystories"), key="lesson_data")
    search = st.text_input("Search lessons", key="lesson_search").strip().lower()

    try:
        lessons = tuple(_plain(lesson) for lesson in list_lessons())
    except (OSError, ValueError) as error:
        st.error(f"Packaged lessons could not be read: {error}")
        return

    matches = []
    for lesson in lessons:
        record = _mapping(lesson)
        searchable = " ".join(
            str(record.get(key, ""))
            for key in ("id", "title", "summary", "mechanism_kind")
        ).lower()
        if not search or search in searchable:
            matches.append(record)
    if not matches:
        st.info("No lessons match this search.")
        return

    for lesson in matches:
        identifier = str(lesson.get("id", "lesson"))
        title = str(lesson.get("title", identifier))
        with st.expander(title, expanded=bool(search)):
            st.text(str(lesson.get("summary", "")))
            st.caption(
                f"Kind: {lesson.get('mechanism_kind', 'unavailable')} · "
                f"Scope: {_text_list(lesson.get('benefit_scope'))}"
            )
            steps = _rows(lesson.get("steps"))
            if steps:
                st.subheader("Shape walkthrough")
                st.dataframe(
                    [
                        {
                            "explanation": step.get("explanation", ""),
                            "shapes": step.get("shapes", ""),
                            "observe": step.get("observe", ""),
                            "source": f"{step.get('source_path', '')}:{step.get('source_symbol', '')}",
                        }
                        for step in steps
                    ],
                    hide_index=True,
                    width="stretch",
                )
            sources = [
                f"{step.get('source_path', '')}:{step.get('source_symbol', '')}"
                for step in steps
                if step.get("source_path")
            ]
            if sources:
                st.text("Implementation sources\n" + "\n".join(sources))
            if lesson.get("try_changes"):
                st.text("Try changing\n" + _text_list(lesson["try_changes"]))
            if lesson.get("limits"):
                st.text("Limits\n" + _text_list(lesson["limits"]))
            st.text(
                "Copyable standalone commands\n"
                + "\n".join(_lesson_commands(identifier, scale, data))
            )


def discover_reports(
    reports_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Load at most 200 direct, content-addressed report bundles.

    The shared loader validates the manifest and all bundle contents. We do not
    inspect report-controlled links, HTML, or Markdown here.
    """
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    try:
        children = sorted(reports_dir.iterdir(), key=lambda item: item.name)[
            :_MAX_BUNDLES
        ]
    except FileNotFoundError:
        return accepted, rejected
    except OSError as error:
        return accepted, [{"bundle": str(reports_dir), "reason": str(error)}]

    for child in children:
        if not child.is_dir() or child.is_symlink():
            continue
        if not _BUNDLE_NAME.fullmatch(child.name):
            rejected.append(
                {
                    "bundle": child.name,
                    "reason": "not a content-addressed bundle directory",
                }
            )
            continue
        manifest = child / "manifest.json"
        if not manifest.is_file() or manifest.is_symlink():
            rejected.append(
                {"bundle": child.name, "reason": "missing safe manifest.json"}
            )
            continue
        try:
            bundle = _mapping(load_report_bundle(child))
            if not bundle:
                raise ValueError("bundle reader returned no report")
            accepted.append({"directory": child.name, "bundle": bundle})
        except (OSError, ValueError, TypeError) as error:
            rejected.append({"bundle": child.name, "reason": str(error)})
    return accepted, rejected


def _report(record: dict[str, object]) -> dict[str, object]:
    bundle = _mapping(record.get("bundle"))
    return _mapping(bundle.get("report", bundle.get("report.json", bundle)))


def _research_entry(report: dict[str, object]) -> dict[str, object]:
    research = _mapping(report.get("research"))
    metadata = _mapping(research.get("metadata", research))
    return _mapping(metadata.get("entry", metadata))


def _report_title(report: dict[str, object]) -> str:
    entry = _research_entry(report)
    return str(entry.get("title") or entry.get("id") or "Unlinked historical evidence")


def _table(title: str, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if materialized:
        st.subheader(title)
        st.dataframe(pd.DataFrame(materialized), hide_index=True, width="stretch")


def _comparison_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for comparison in _rows(report.get("comparisons")):
        capabilities = _mapping(comparison.get("capabilities"))
        for card, result in capabilities.items():
            detail = _mapping(result)
            delta = detail.get("score_delta", detail.get("delta"))
            if isinstance(delta, dict):
                delta = delta.get("value")
            rows.append(
                {
                    "comparison": comparison.get(
                        "id", comparison.get("name", "comparison")
                    ),
                    "card": card,
                    "seed": detail.get("seed", comparison.get("seed", "")),
                    "observed_delta": delta,
                    "status": detail.get("status", comparison.get("status", "")),
                    "reason": detail.get("reason", comparison.get("reason", "")),
                }
            )
    return rows


def _capability_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for run in _rows(report.get("runs")):
        run_id = run.get("run_id", "")
        endpoint = _mapping(run.get("endpoint_status"))
        for card, capability in _mapping(run.get("capabilities")).items():
            detail = _mapping(capability)
            outcome = _mapping(detail.get("result", detail))
            cases = _rows(outcome.get("results"))
            if not cases:
                rows.append(
                    {
                        "run_id": run_id,
                        "endpoint_status": endpoint.get("status", "unavailable"),
                        "card": card,
                        "valid": outcome.get("valid", False),
                        "error": outcome.get("error", "case evidence unavailable"),
                    }
                )
            for case in cases:
                rows.append(
                    {
                        "run_id": run_id,
                        "endpoint_status": endpoint.get("status", "unavailable"),
                        "card": card,
                        **case,
                    }
                )
    return rows


def _card_score_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for run in _rows(report.get("runs")):
        for card, capability in _mapping(run.get("capabilities")).items():
            detail = _mapping(capability)
            outcome = _mapping(detail.get("result", detail))
            score = outcome.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                rows.append(
                    {"run_id": run.get("run_id", ""), "card": card, "score": score}
                )
    return rows


def _render_bundle(record: dict[str, object]) -> None:
    report = _report(record)
    st.subheader(_report_title(report))
    st.caption(f"Bundle: {record['directory']}")
    entry = _research_entry(report)
    if not entry:
        st.warning(
            "Unlinked historical evidence: hypothesis metadata was not recorded."
        )
    else:
        for field, heading in (
            ("question", "Question"),
            ("hypothesis", "Hypothesis"),
            ("failure_interpretation", "Failure interpretation"),
        ):
            if entry.get(field):
                st.text(f"{heading}\n{entry[field]}")
        _table(
            "Controls and varied fields",
            [
                {
                    "fixed controls": _text_list(entry.get("controls")),
                    "varied fields": _text_list(entry.get("independent_variables")),
                }
            ],
        )
        _table("Cards", _rows(entry.get("cards")))
    research = _mapping(report.get("research"))
    if research.get("dataset_profile") is not None:
        st.subheader("Dataset and resource provenance")
        st.json(research["dataset_profile"])
    st.subheader("Architectural quantities and cache templates")
    st.json(report.get("architectural_quantities", []))
    st.subheader("Implementation telemetry")
    st.json(report.get("implementation_quantities", {}))
    st.subheader("Measured costs and unavailable values")
    st.json(report.get("costs", {}))
    st.subheader("Local evidence validation")
    st.json(report.get("local_validation", {}))

    comparisons = _comparison_rows(report)
    _table("Observed pair deltas", comparisons)
    numeric = [
        row
        for row in comparisons
        if isinstance(row.get("observed_delta"), (int, float))
        and not isinstance(row.get("observed_delta"), bool)
    ]
    if numeric:
        st.plotly_chart(
            px.bar(
                pd.DataFrame(numeric),
                x="comparison",
                y="observed_delta",
                color="card",
                hover_data=["seed", "status"],
            ),
            width="stretch",
        )
    capabilities = _capability_rows(report)
    _table("Raw card outcomes", capabilities)
    score_rows = _card_score_rows(report)
    if score_rows:
        st.plotly_chart(
            px.strip(
                pd.DataFrame(score_rows),
                x="card",
                y="score",
                color="card",
                hover_data=["run_id"],
            ),
            width="stretch",
        )
    _table(
        "Endpoint status",
        [
            {"run_id": run.get("run_id", ""), **_mapping(run.get("endpoint_status"))}
            for run in _rows(report.get("runs"))
        ],
    )
    _table("Missing or rejected evidence", _rows(report.get("missing_or_rejected")))
    _table(
        "Limitations",
        [{"limitation": item} for item in report.get("limitations", [])]
        if isinstance(report.get("limitations"), list)
        else [],
    )


def research_page(reports_dir: Path) -> None:
    """Render declarative research records and validated immutable reports."""
    st.header("Research")
    st.write(
        "Browse declared hypotheses and immutable evidence bundles. Reports are read-only; no run database is opened."
    )
    try:
        entries = [_mapping(_plain(entry)) for entry in list_research()]
    except (OSError, ValueError) as error:
        st.error(f"Packaged research catalog could not be read: {error}")
        entries = []
    for entry in entries:
        with st.expander(str(entry.get("title", entry.get("id", "Research entry")))):
            for field, title in (
                ("question", "Question"),
                ("hypothesis", "Hypothesis"),
                ("failure_interpretation", "Failure interpretation"),
            ):
                if entry.get(field):
                    st.text(f"{title}\n{entry[field]}")
            _table(
                "Controls and varied fields",
                [
                    {
                        "fixed controls": _text_list(entry.get("controls")),
                        "varied fields": _text_list(entry.get("independent_variables")),
                    }
                ],
            )
            _table("Dataset/card applicability", _rows(entry.get("cards")))
            st.text(
                f"Recommended scale: {entry.get('recommended_scale', 'unavailable')}\nResource class: {entry.get('runtime_class', 'unavailable')}"
            )
            for paper in _rows(entry.get("papers")):
                url = paper.get("url")
                if isinstance(url, str) and url.startswith("https://"):
                    st.link_button(str(paper.get("title", url)), url)

    st.subheader("Completed reports")
    bundles, rejected = discover_reports(reports_dir)
    st.caption(
        f"Report root: {reports_dir} (direct children only; maximum {_MAX_BUNDLES})"
    )
    if rejected:
        st.warning("Some report bundles were rejected and were not rendered.")
        st.dataframe(rejected, hide_index=True, width="stretch")
    if not bundles:
        st.info("No validated report bundles are available.")
    for bundle in bundles:
        with st.expander(_report_title(_report(bundle))):
            _render_bundle(bundle)
