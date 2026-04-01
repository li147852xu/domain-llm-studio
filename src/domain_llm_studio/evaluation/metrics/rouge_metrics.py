"""ROUGE metrics for summarization tasks."""

from __future__ import annotations

from rouge_score import rouge_scorer


def compute_rouge(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    scores = {"rouge1": [], "rouge2": [], "rougeL": []}
    for pred, ref in zip(predictions, references):
        result = scorer.score(ref, pred)
        for key in scores:
            scores[key].append(result[key].fmeasure)

    return {k: sum(v) / len(v) if v else 0.0 for k, v in scores.items()}


def compute_bertscore(
    predictions: list[str], references: list[str], lang: str = "en"
) -> dict[str, float]:
    """Compute BERTScore F1."""
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(predictions, references, lang=lang, verbose=False)
        return {
            "bertscore_precision": P.mean().item(),
            "bertscore_recall": R.mean().item(),
            "bertscore_f1": F1.mean().item(),
        }
    except Exception:
        return {"bertscore_precision": 0.0, "bertscore_recall": 0.0, "bertscore_f1": 0.0}


def compute_keypoint_coverage(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute keyword/keypoint coverage ratio.

    Measures what fraction of key terms in the reference appear in the prediction.
    Uses simple word overlap as a proxy.
    """
    import re

    coverage_scores = []
    for pred, ref in zip(predictions, references):
        ref_words = set(re.findall(r"\w+", ref.lower()))
        pred_words = set(re.findall(r"\w+", pred.lower()))

        if not ref_words:
            coverage_scores.append(1.0)
            continue

        stopwords = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
                      "to", "for", "of", "and", "or", "with", "的", "了", "在", "是",
                      "和", "有", "为", "与", "及", "等", "null", "none"}
        ref_keywords = ref_words - stopwords
        if not ref_keywords:
            coverage_scores.append(1.0)
            continue

        covered = ref_keywords & pred_words
        coverage_scores.append(len(covered) / len(ref_keywords))

    return {"keypoint_coverage": sum(coverage_scores) / len(coverage_scores) if coverage_scores else 0.0}
