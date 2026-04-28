"""Convert ResearchOps research runs into 4-task SFT samples — the data flywheel.

A *ResearchOps run* is the artifact dump of a multi-step research workflow on
a single research question. It contains, at minimum:

- ``plan.json``        — the research question + decomposed sub-questions
- ``sources.jsonl``    — sources gathered during research, one JSON per line:
                         ``{title, url, published_at, body, [events]}``
- ``report.md``        — the final markdown report (with inline citations)
- ``evidence_map.json``— ``{question_id: {"answer": str, "evidence": str,
                                          "source_url": str}}``

This importer emits up to 4 kinds of SFT samples per run:

- ``fin_summary``       — quoted paragraphs from ``report.md``  → ``(doc, summary, key_points)``
- ``event_extraction``  — events embedded in ``sources.jsonl``  → entity/event objects
- ``doc_qa``            — research questions in ``plan.json`` + ``evidence_map`` → ``(q, evidence, answer)``
- ``analysis_gen``      — numeric-fact paragraphs in ``report.md`` → structured analysis input/output

If a run is missing one of those source files, that task type is **skipped
without raising** — partial runs still contribute whatever they can.

Usage
-----

CLI::

    python -m domain_llm_studio.data.researchops_importer \\
        --runs-dir /path/to/researchops/runs \\
        --output data/processed/from_researchops

Programmatic::

    from domain_llm_studio.data.researchops_importer import import_runs
    import_runs(runs_dir, output_dir)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

_NUMERIC_PATTERN = re.compile(
    r"\b(\$?[0-9]{1,3}(?:[\,\.][0-9]{2,3})*(?:[KMB]|%)?|\d+%|\d+\.\d+%?)"
)
_HEADING_PATTERN = re.compile(r"^#+\s")


def _detect_lang(text: str) -> str:
    return "zh" if any("\u4e00" <= c <= "\u9fff" for c in text) else "en"


def _split_paragraphs(md: str) -> list[str]:
    """Split markdown into paragraphs, stripping headings + blank lines."""
    out: list[str] = []
    for chunk in md.split("\n\n"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if _HEADING_PATTERN.match(chunk):
            continue
        out.append(chunk)
    return out


def _extract_key_points(paragraph: str, max_points: int = 4) -> list[str]:
    """Coarse keypoint extractor: pick sentences that contain a number."""
    sentences = re.split(r"(?<=[\.\!\?。！？])\s+", paragraph)
    points: list[str] = []
    for s in sentences:
        s = s.strip()
        if _NUMERIC_PATTERN.search(s):
            points.append(s)
            if len(points) >= max_points:
                break
    return points


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Per-task converters
# ---------------------------------------------------------------------------

def make_fin_summary_samples(report_md: str, max_per_run: int = 5) -> list[dict]:
    paragraphs = _split_paragraphs(report_md)
    samples: list[dict] = []
    for para in paragraphs:
        if len(para) < 80:
            continue
        keypoints = _extract_key_points(para)
        if not keypoints:
            continue
        lang = _detect_lang(para)
        summary = keypoints[0] if keypoints else para[:120]
        output = {
            "summary": summary,
            "key_points": keypoints,
            "risks": [],
            "opportunities": [],
        }
        samples.append({
            "task": "fin_summary",
            "lang": lang,
            "input": para,
            "output": json.dumps(output, ensure_ascii=False),
        })
        if len(samples) >= max_per_run:
            break
    return samples


def make_event_extraction_samples(
    sources: list[dict], max_per_run: int = 5
) -> list[dict]:
    samples: list[dict] = []
    for src in sources:
        events = src.get("events") or []
        if not events:
            continue
        body = src.get("body") or src.get("title") or ""
        if not body:
            continue
        lang = _detect_lang(body)
        samples.append({
            "task": "event_extraction",
            "lang": lang,
            "input": body,
            "output": json.dumps(events, ensure_ascii=False),
        })
        if len(samples) >= max_per_run:
            break
    return samples


def make_doc_qa_samples(
    plan: dict, evidence_map: dict, max_per_run: int = 5
) -> list[dict]:
    """Marry plan questions with evidence_map answers/spans."""
    samples: list[dict] = []
    questions = plan.get("questions") or plan.get("sub_questions") or []
    for q in questions:
        qid = q.get("id") if isinstance(q, dict) else None
        question_text = (
            q.get("text") if isinstance(q, dict) else (q if isinstance(q, str) else None)
        )
        if not question_text or qid is None:
            continue
        evidence_entry = evidence_map.get(qid) or evidence_map.get(str(qid))
        if not evidence_entry:
            continue
        answer = evidence_entry.get("answer")
        evidence = evidence_entry.get("evidence")
        if not answer or not evidence:
            continue
        lang = _detect_lang(question_text + " " + evidence)
        qa_input = json.dumps(
            {"context": evidence, "question": question_text},
            ensure_ascii=False,
        )
        qa_output = json.dumps(
            {"answer": answer, "evidence_span": evidence},
            ensure_ascii=False,
        )
        samples.append({
            "task": "doc_qa",
            "lang": lang,
            "input": qa_input,
            "output": qa_output,
        })
        if len(samples) >= max_per_run:
            break
    return samples


def make_analysis_gen_samples(
    report_md: str, plan: dict, max_per_run: int = 3
) -> list[dict]:
    """Turn numeric paragraphs into (structured_data → memo paragraph) pairs."""
    paragraphs = _split_paragraphs(report_md)
    company = plan.get("company") or plan.get("subject") or "Subject"
    period = plan.get("period") or plan.get("time_window") or "recent period"

    samples: list[dict] = []
    for para in paragraphs:
        nums = _NUMERIC_PATTERN.findall(para)
        if len(nums) < 2:
            continue
        lang = _detect_lang(para)
        structured = {
            "company": company,
            "period": period,
            "key_numbers": nums[:5],
            "snippet": para[:160],
        }
        samples.append({
            "task": "analysis_gen",
            "lang": lang,
            "input": json.dumps(structured, ensure_ascii=False),
            "output": para,
        })
        if len(samples) >= max_per_run:
            break
    return samples


# ---------------------------------------------------------------------------
# Per-run + driver
# ---------------------------------------------------------------------------

def import_run(run_dir: Path) -> list[dict]:
    """Return all samples derivable from a single run directory."""
    samples: list[dict] = []

    plan_path = run_dir / "plan.json"
    sources_path = run_dir / "sources.jsonl"
    report_path = run_dir / "report.md"
    evidence_path = run_dir / "evidence_map.json"

    plan: dict = _read_json(plan_path) if plan_path.exists() else {}
    sources: list[dict] = _read_jsonl(sources_path) if sources_path.exists() else []
    report_md: str = _read_text(report_path) if report_path.exists() else ""
    evidence_map: dict = _read_json(evidence_path) if evidence_path.exists() else {}

    if report_md:
        samples += make_fin_summary_samples(report_md)
    if sources:
        samples += make_event_extraction_samples(sources)
    if plan and evidence_map:
        samples += make_doc_qa_samples(plan, evidence_map)
    if report_md and plan:
        samples += make_analysis_gen_samples(report_md, plan)

    for s in samples:
        s["source_run"] = run_dir.name

    return samples


def _split(samples: list[dict], seed: int = 42) -> tuple[list[dict], list[dict], list[dict]]:
    """Stratified-by-task 80/10/10 split."""
    import random

    rng = random.Random(seed)
    by_task: dict[str, list[dict]] = {}
    for s in samples:
        by_task.setdefault(s["task"], []).append(s)

    train: list[dict] = []
    dev: list[dict] = []
    test: list[dict] = []
    for task_samples in by_task.values():
        rng.shuffle(task_samples)
        n = len(task_samples)
        n_train = max(1, int(n * 0.8))
        n_dev = max(0, int(n * 0.1))
        train += task_samples[:n_train]
        dev += task_samples[n_train : n_train + n_dev]
        test += task_samples[n_train + n_dev :]
    return train, dev, test


def import_runs(
    runs_dir: Path,
    output_dir: Path,
    seed: int = 42,
) -> dict[str, int]:
    """Import all runs under ``runs_dir`` and write 4-task split JSONL files."""
    runs_dir = Path(runs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not runs_dir.exists():
        raise FileNotFoundError(f"runs_dir does not exist: {runs_dir}")

    all_samples: list[dict] = []
    n_runs = 0
    for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        run_samples = import_run(run_dir)
        all_samples.extend(run_samples)
        n_runs += 1
        logger.info(
            "Run %s: produced %d samples across %d tasks",
            run_dir.name,
            len(run_samples),
            len({s["task"] for s in run_samples}),
        )

    train, dev, test = _split(all_samples, seed=seed)

    counts = {"train": len(train), "dev": len(dev), "test": len(test)}
    for split_name, split_samples in [("train", train), ("dev", dev), ("test", test)]:
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for s in split_samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(
        "Imported %d runs → %d train / %d dev / %d test samples",
        n_runs, counts["train"], counts["dev"], counts["test"],
    )
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/from_researchops"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    counts = import_runs(args.runs_dir, args.output, seed=args.seed)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
