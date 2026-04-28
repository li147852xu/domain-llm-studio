# Data Flywheel Demo (toy scale)

- ResearchOps runs imported from: `tests/fixtures/researchops_run`
- Imported samples: {'train': 12, 'dev': 0, 'test': 5}
- Merged splits (toy snapshot 16 base + ResearchOps): {'train': 28, 'dev': 16, 'test': 21}
- Mini LoRA: r=8, 1 epoch on 28 samples

## Eval (n=4 samples per task, deterministic)

| Metric | base | flywheel_tuned | Δ |
|---|---|---|---|
| ROUGE-L (avg over fin_summary etc.) | 0.5733977801646976 | 0.6145972407654651 | +0.0412 |
| grounding_rate (doc_qa) | 1.0 | 1.0 | +0.0000 |

## Timeline (wall-clock)

| Step | Seconds |
|---|---|
| 1. researchops_importer | 0.1 |
| 3. mini LoRA training (1.5B) | 20.3 |
| 4a. eval base | 13.0 |
| 4b. eval flywheel_tuned | 16.8 |

## TL;DR

Imported 17 samples from 3 ResearchOps runs and trained a toy-scale LoRA on top of 65 merged samples; this demonstrates the data-flywheel loop end-to-end.

