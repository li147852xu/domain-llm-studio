# Evaluation Report

*Generated: 2026-04-03 12:45:49*

## Model Variants

- **BASE**
- **PROMPT_ONLY**
- **TUNED**

## Per-Task Results

### analysis_gen

| Metric | BASE | PROMPT_ONLY | TUNED | Delta |
|--------|--------|--------|--------|--------|
| field_completeness | 0.8667 | 0.8000 | 1.0000 | +0.1333 |
| format_compliance | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| schema_match | 0.9333 | 0.9000 | 1.0000 | +0.0667 |

### doc_qa

| Metric | BASE | PROMPT_ONLY | TUNED | Delta |
|--------|--------|--------|--------|--------|
| exact_match | 0.8000 | 0.8000 | 1.0000 | +0.2000 |
| grounding_rate | 0.9000 | 1.0000 | 1.0000 | +0.1000 |
| token_f1 | 0.9556 | 0.9556 | 1.0000 | +0.0444 |

### event_extraction

| Metric | BASE | PROMPT_ONLY | TUNED | Delta |
|--------|--------|--------|--------|--------|
| entity_f1 | 0.9000 | 0.9333 | 1.0000 | +0.1000 |
| entity_match_rate | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| entity_precision | 0.8500 | 0.9000 | 1.0000 | +0.1500 |
| entity_recall | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| event_f1 | 0.0667 | 0.3667 | 0.4000 | +0.3333 |
| event_precision | 0.0500 | 0.3500 | 0.4000 | +0.3500 |
| event_recall | 0.1000 | 0.4000 | 0.4000 | +0.3000 |
| key_presence_rate | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| parse_failure_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| partial_field_match | 0.8500 | 0.8833 | 0.9000 | +0.0500 |

### fin_summary

| Metric | BASE | PROMPT_ONLY | TUNED | Delta |
|--------|--------|--------|--------|--------|
| keypoint_coverage | 0.6499 | 0.5850 | 0.7927 | +0.1429 |
| rouge1 | 0.7061 | 0.7255 | 0.9165 | +0.2104 |
| rouge2 | 0.5066 | 0.5490 | 0.9002 | +0.3936 |
| rougeL | 0.6328 | 0.6820 | 0.9165 | +0.2837 |

## Error Analysis

### BASE

- Error rate: 100.0%
- Total errors: 40

**Error Distribution:**

- partial_match: 35
- hallucination: 5

**Example Failures:**

*hallucination* (task: fin_summary):
  - Input: `万科A发布了2025年Q3季度业绩报告。报告期内,公司实现营业收入1479.7亿元,同比下降34.3%。归属于上市公司股东的净利润为3292.9亿元,营业收入为10.7%。公司核心业务保持稳健增长,但...`
  - Prediction: `{
  "summary": "万科A在2025年第三季度实现了1479.7亿元的营业收入，同比下降34.3%，净利润为3292.9亿元，同比增长10.7%，尽管面临消费需求放缓的挑战，但核心业务保持...`
  - Reference: `{"summary": "万科A2025年Q3季度实现营收1479.7亿元,同比下降34.3%。净利润为3292.9亿元。公司核心业务表现稳健。", "key_points": ["营业收入1479....`

*partial_match* (task: analysis_gen):
  - Input: `{"company": "海天味业", "period": "2025年Q3季度", "revenue": "2716.0亿元", "yoy_growth": "-17.9%", "net_incom...`
  - Prediction: `海天味业在2025年第三季度的财务报告显示，公司实现营业收入2716.0亿元，较去年同期下降17.9%，净利润为2835.7亿元，毛利率维持在19.7%。其中，消费板块收入下降了17.2%，是影响整体...`
  - Reference: `海天味业2025年Q3季度实现营业收入2716.0亿元,同比下滑17.9%。净利润达到2835.7亿元,利润率为19.7%。消费板块收入下降17.2%,面临一定经营压力。展望未来,成本优化措施有望帮助...`


### PROMPT_ONLY

- Error rate: 92.5%
- Total errors: 37

**Error Distribution:**

- partial_match: 32
- hallucination: 5

**Example Failures:**

*hallucination* (task: fin_summary):
  - Input: `万科A发布了2025年Q3季度业绩报告。报告期内,公司实现营业收入1479.7亿元,同比下降34.3%。归属于上市公司股东的净利润为3292.9亿元,营业收入为10.7%。公司核心业务保持稳健增长,但...`
  - Prediction: `{
  "summary": "万科A在2025年第三季度实现营业收入1479.7亿元，同比下降34.3%，净利润为3292.9亿元，同比下降10.7%。",
  "key_points": [
  ...`
  - Reference: `{"summary": "万科A2025年Q3季度实现营收1479.7亿元,同比下降34.3%。净利润为3292.9亿元。公司核心业务表现稳健。", "key_points": ["营业收入1479....`

*partial_match* (task: analysis_gen):
  - Input: `{"company": "海天味业", "period": "2025年Q3季度", "revenue": "2716.0亿元", "yoy_growth": "-17.9%", "net_incom...`
  - Prediction: `海天味业在2025年第三季度实现营业收入2716.0亿元，同比下滑17.9%，净利润为2835.7亿元，毛利率保持在19.7%。消费板块收入下降17.2%，表明市场需求有所减弱。展望未来，公司需进一步...`
  - Reference: `海天味业2025年Q3季度实现营业收入2716.0亿元,同比下滑17.9%。净利润达到2835.7亿元,利润率为19.7%。消费板块收入下降17.2%,面临一定经营压力。展望未来,成本优化措施有望帮助...`


### TUNED

- Error rate: 52.5%
- Total errors: 21

**Error Distribution:**

- partial_match: 16
- hallucination: 5

**Example Failures:**

*hallucination* (task: fin_summary):
  - Input: `万科A发布了2025年Q3季度业绩报告。报告期内,公司实现营业收入1479.7亿元,同比下降34.3%。归属于上市公司股东的净利润为3292.9亿元,营业收入为10.7%。公司核心业务保持稳健增长,但...`
  - Prediction: `{"summary": "万科A发布2025年Q3季度业绩报告,实现营收1479.7亿元,同比下降34.3%。净利润为3292.9亿元。公司核心业务表现稳健。", "key_points": ["营业...`
  - Reference: `{"summary": "万科A2025年Q3季度实现营收1479.7亿元,同比下降34.3%。净利润为3292.9亿元。公司核心业务表现稳健。", "key_points": ["营业收入1479....`

*partial_match* (task: analysis_gen):
  - Input: `{"company": "海天味业", "period": "2025年Q3季度", "revenue": "2716.0亿元", "yoy_growth": "-17.9%", "net_incom...`
  - Prediction: `海天味业2025年Q3季度实现营业收入2716.0亿元,同比下降17.9%。净利润达到2835.7亿元,利润率为19.7%。消费板块收入下降17.2%,表明成本优化措施可能尚未完全见效。展望未来,持续...`
  - Reference: `海天味业2025年Q3季度实现营业收入2716.0亿元,同比下滑17.9%。净利润达到2835.7亿元,利润率为19.7%。消费板块收入下降17.2%,面临一定经营压力。展望未来,成本优化措施有望帮助...`

