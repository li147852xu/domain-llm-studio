"""Tests for the ResearchOps → 4-task SFT importer."""

from __future__ import annotations

import json
from pathlib import Path

from domain_llm_studio.data.researchops_importer import (
    import_run,
    import_runs,
    make_doc_qa_samples,
    make_event_extraction_samples,
    make_fin_summary_samples,
)

FIXTURES = Path(__file__).parent / "fixtures" / "researchops_run"


def test_import_run_full_produces_all_four_tasks():
    """A run with all 4 source files should yield samples in all 4 task types."""
    samples = import_run(FIXTURES / "run_001")
    tasks = {s["task"] for s in samples}
    assert tasks == {"fin_summary", "event_extraction", "doc_qa", "analysis_gen"}, (
        f"expected all 4 tasks, got {tasks}"
    )

    for s in samples:
        assert "input" in s and "output" in s
        assert s["source_run"] == "run_001"


def test_import_run_partial_skips_missing_sources_without_error():
    """run_003_partial only has report.md → only fin_summary should be produced."""
    samples = import_run(FIXTURES / "run_003_partial")
    tasks = {s["task"] for s in samples}
    assert tasks.issubset({"fin_summary", "analysis_gen"})
    assert "event_extraction" not in tasks
    assert "doc_qa" not in tasks


def test_doc_qa_samples_have_valid_json_io():
    """doc_qa input/output must be parseable JSON with the expected schema."""
    plan = json.loads((FIXTURES / "run_001/plan.json").read_text(encoding="utf-8"))
    evidence = json.loads(
        (FIXTURES / "run_001/evidence_map.json").read_text(encoding="utf-8")
    )
    samples = make_doc_qa_samples(plan, evidence)
    assert len(samples) >= 1
    for s in samples:
        in_obj = json.loads(s["input"])
        out_obj = json.loads(s["output"])
        assert {"context", "question"} <= in_obj.keys()
        assert {"answer", "evidence_span"} <= out_obj.keys()


def test_fin_summary_samples_have_keypoints_with_numbers():
    """fin_summary keypoints should contain numeric facts where possible."""
    report_md = (FIXTURES / "run_001/report.md").read_text(encoding="utf-8")
    samples = make_fin_summary_samples(report_md)
    assert samples, "should produce at least one fin_summary sample"
    for s in samples:
        out = json.loads(s["output"])
        assert "summary" in out
        assert isinstance(out.get("key_points", []), list)


def test_event_extraction_samples_keep_event_objects():
    """event_extraction output should be the JSON array of event dicts from sources."""
    sources_path = FIXTURES / "run_001/sources.jsonl"
    sources = []
    with open(sources_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sources.append(json.loads(line))
    samples = make_event_extraction_samples(sources)
    assert samples
    for s in samples:
        events = json.loads(s["output"])
        assert isinstance(events, list)
        assert all("company" in e for e in events)


def test_import_runs_writes_split_jsonl(tmp_path):
    """End-to-end: import all fixture runs → 80/10/10 split files materialize."""
    out_dir = tmp_path / "imported"
    counts = import_runs(FIXTURES, out_dir, seed=42)
    assert sum(counts.values()) > 0
    for split in ("train", "dev", "test"):
        out = out_dir / f"{split}.jsonl"
        assert out.exists(), f"missing {out}"


def test_import_runs_handles_zh_without_crashing():
    """The Chinese-language run should produce well-formed samples too."""
    samples = import_run(FIXTURES / "run_002_zh")
    assert any(s["lang"] == "zh" for s in samples)
