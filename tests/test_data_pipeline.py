"""Tests for the data pipeline: seed generation, cleaning, formatting, splitting."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from domain_llm_studio.data.cleaners import (
    clean_dataset,
    clean_sample,
    content_hash,
    deduplicate,
    normalize_whitespace,
    strip_html,
)
from domain_llm_studio.data.formatters import format_sample
from domain_llm_studio.data.seed_generator import generate_seed_data
from domain_llm_studio.data.splitter import stratified_split
from domain_llm_studio.data.stats import compute_stats


class TestSeedGenerator:
    def test_generates_all_tasks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            counts = generate_seed_data(Path(tmpdir), num_samples_per_task=5, seed=42)
            assert len(counts) == 4
            for task, count in counts.items():
                assert count == 10  # 5 en + 5 zh

    def test_reproducible_with_seed(self):
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            generate_seed_data(Path(d1), num_samples_per_task=3, seed=42)
            generate_seed_data(Path(d2), num_samples_per_task=3, seed=42)
            for f in Path(d1).glob("*.jsonl"):
                content1 = f.read_text()
                content2 = (Path(d2) / f.name).read_text()
                assert content1 == content2

    def test_bilingual_output(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            generate_seed_data(Path(tmpdir), num_samples_per_task=3, seed=42)
            all_samples = []
            for f in Path(tmpdir).glob("*.jsonl"):
                with open(f) as fp:
                    for line in fp:
                        all_samples.append(json.loads(line))
            langs = {s["lang"] for s in all_samples}
            assert "en" in langs
            assert "zh" in langs


class TestCleaners:
    def test_normalize_whitespace(self):
        assert normalize_whitespace("  hello   world  ") == "hello world"

    def test_strip_html(self):
        assert strip_html("<b>bold</b> text") == "bold text"

    def test_content_hash_deterministic(self):
        assert content_hash("test") == content_hash("test")
        assert content_hash("test") != content_hash("other")

    def test_deduplicate(self):
        samples = [
            {"input": "same text", "output": "a"},
            {"input": "same text", "output": "b"},
            {"input": "different", "output": "c"},
        ]
        result = deduplicate(samples)
        assert len(result) == 2

    def test_clean_sample_filters_empty(self):
        assert clean_sample({"input": "", "output": "text"}) is None
        assert clean_sample({"input": "text", "output": ""}) is None

    def test_clean_dataset_stats(self):
        samples = [
            {"input": "hello", "output": "world", "task": "analysis_gen"},
            {"input": "hello", "output": "world", "task": "analysis_gen"},
            {"input": "unique", "output": "data", "task": "analysis_gen"},
        ]
        cleaned, stats = clean_dataset(samples)
        assert stats["original"] == 3
        assert stats["after_dedup"] == 2
        assert stats["retention_rate"] < 1.0


class TestFormatters:
    def test_format_sample(self):
        sample = {
            "task": "fin_summary",
            "lang": "en",
            "input": "Some financial document",
            "output": '{"summary": "test"}',
        }
        result = format_sample(sample)
        assert "instruction" in result
        assert result["input"] == "Some financial document"
        assert result["task"] == "fin_summary"

    def test_format_qa_sample(self):
        sample = {
            "task": "doc_qa",
            "lang": "en",
            "input": json.dumps({"context": "Tesla delivered 100K cars", "question": "How many?"}),
            "output": '{"answer": "100K"}',
        }
        result = format_sample(sample)
        assert "Context:" in result["input"]
        assert "Question:" in result["input"]


class TestSplitter:
    def test_split_ratios(self):
        samples = [{"task": "a", "input": str(i)} for i in range(100)]
        train, dev, test = stratified_split(samples, seed=42)
        assert len(train) == 80
        assert len(dev) == 10
        assert len(test) == 10

    def test_split_reproducible(self):
        samples = [{"task": "a", "input": str(i)} for i in range(50)]
        t1, d1, _ = stratified_split(samples, seed=42)
        t2, d2, _ = stratified_split(samples, seed=42)
        assert [s["input"] for s in t1] == [s["input"] for s in t2]


class TestStats:
    def test_compute_stats(self):
        samples = [
            {"task": "fin_summary", "lang": "en", "input": "hello", "output": "world"},
            {"task": "doc_qa", "lang": "zh", "input": "test input", "output": "test out"},
        ]
        stats = compute_stats(samples)
        assert stats["total"] == 2
        assert "fin_summary" in stats["by_task"]
        assert "en" in stats["by_lang"]
