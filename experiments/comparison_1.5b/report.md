# Evaluation Report

*Generated: 2026-04-03 11:41:19*

## Model Variants

- **BASE**
- **PROMPT_ONLY**
- **TUNED**

## Per-Task Results

### analysis_gen

| Metric | BASE | PROMPT_ONLY | TUNED | Delta |
|--------|--------|--------|--------|--------|
| field_completeness | 0.5000 | 0.7000 | 1.0000 | +0.5000 |
| format_compliance | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| schema_match | 0.7500 | 0.8500 | 1.0000 | +0.2500 |

### doc_qa

| Metric | BASE | PROMPT_ONLY | TUNED | Delta |
|--------|--------|--------|--------|--------|
| exact_match | 0.8000 | 0.8000 | 1.0000 | +0.2000 |
| grounding_rate | 0.8000 | 0.8000 | 1.0000 | +0.2000 |
| token_f1 | 0.9167 | 0.8000 | 1.0000 | +0.0833 |

### event_extraction

| Metric | BASE | PROMPT_ONLY | TUNED | Delta |
|--------|--------|--------|--------|--------|
| entity_f1 | 0.8000 | 0.8667 | 1.0000 | +0.2000 |
| entity_match_rate | 0.9000 | 1.0000 | 1.0000 | +0.1000 |
| entity_precision | 0.7500 | 0.8000 | 1.0000 | +0.2500 |
| entity_recall | 0.9000 | 1.0000 | 1.0000 | +0.1000 |
| event_f1 | 0.1667 | 0.1667 | 0.6000 | +0.4333 |
| event_precision | 0.1500 | 0.1500 | 0.6000 | +0.4500 |
| event_recall | 0.2000 | 0.2000 | 0.6000 | +0.4000 |
| key_presence_rate | 1.0000 | 1.0000 | 1.0000 | +0.0000 |
| parse_failure_rate | 0.0000 | 0.0000 | 0.0000 | +0.0000 |
| partial_field_match | 0.7167 | 0.6333 | 0.9333 | +0.2167 |

### fin_summary

| Metric | BASE | PROMPT_ONLY | TUNED | Delta |
|--------|--------|--------|--------|--------|
| keypoint_coverage | 0.5452 | 0.5120 | 0.8145 | +0.2693 |
| rouge1 | 0.6582 | 0.6670 | 0.9276 | +0.2694 |
| rouge2 | 0.4747 | 0.4323 | 0.9139 | +0.4391 |
| rougeL | 0.5982 | 0.6284 | 0.9248 | +0.3266 |

## Error Analysis

### BASE

- Error rate: 100.0%
- Total errors: 40

**Error Distribution:**

- partial_match: 28
- format_violation: 10
- grounding_failure: 2

**Example Failures:**

*format_violation* (task: fin_summary):
  - Input: `万科A发布了2025年Q3季度业绩报告。报告期内,公司实现营业收入1479.7亿元,同比下降34.3%。归属于上市公司股东的净利润为3292.9亿元,营业收入为10.7%。公司核心业务保持稳健增长,但...`
  - Prediction: ````json
{
  "summary": "万科A发布2025年第三季度业绩，营收和利润均下降，但核心业务稳健增长。",
  "key_points": [
    "营业收入1479.7亿元，同...`
  - Reference: `{"summary": "万科A2025年Q3季度实现营收1479.7亿元,同比下降34.3%。净利润为3292.9亿元。公司核心业务表现稳健。", "key_points": ["营业收入1479....`

*grounding_failure* (task: doc_qa):
  - Input: `Context: On 2024-12-04, Intel Corp. announced the appointment of Lisa Wang as the new CEO, replacing...`
  - Prediction: `{
  "answer": "The leadership change was made due to the need for fresh leadership to navigate the c...`
  - Reference: `{"answer": "to navigate the company's expansion into electric vehicles", "evidence_span": "to naviga...`

*partial_match* (task: analysis_gen):
  - Input: `{"company": "海天味业", "period": "2025年Q3季度", "revenue": "2716.0亿元", "yoy_growth": "-17.9%", "net_incom...`
  - Prediction: `In the third quarter of 2025, our flagship brand, 海天味业，实现了总营收2716.0亿元，较去年同期下滑了17.9%，显示出市场环境的挑战。尽管整体业...`
  - Reference: `海天味业2025年Q3季度实现营业收入2716.0亿元,同比下滑17.9%。净利润达到2835.7亿元,利润率为19.7%。消费板块收入下降17.2%,面临一定经营压力。展望未来,成本优化措施有望帮助...`


### PROMPT_ONLY

- Error rate: 100.0%
- Total errors: 40

**Error Distribution:**

- partial_match: 33
- hallucination: 5
- grounding_failure: 2

**Example Failures:**

*grounding_failure* (task: doc_qa):
  - Input: `Context: 招商银行于2023-11-13公告,任命王芳为公司新任首席执行官,接替任职6年的李明。董事会表示,此次管理层调整旨在推动公司在金融科技领域的战略转型。

Question: 李明担任...`
  - Prediction: `{"answer": "6 years", "evidence_span": "任命王芳为公司新任首席执行官,接替任职6年的李明"}...`
  - Reference: `{"answer": "6年", "evidence_span": "6年"}...`

*hallucination* (task: fin_summary):
  - Input: `万科A发布了2025年Q3季度业绩报告。报告期内,公司实现营业收入1479.7亿元,同比下降34.3%。归属于上市公司股东的净利润为3292.9亿元,营业收入为10.7%。公司核心业务保持稳健增长,但...`
  - Prediction: `{"summary":"万科A发布2025年Q3业绩报告，营业收入下降34.3%，净利润率为11.7%。","key_points":["营业收入1479.7亿元","净利润3292.9亿元","营业...`
  - Reference: `{"summary": "万科A2025年Q3季度实现营收1479.7亿元,同比下降34.3%。净利润为3292.9亿元。公司核心业务表现稳健。", "key_points": ["营业收入1479....`

*partial_match* (task: analysis_gen):
  - Input: `{"company": "海天味业", "period": "2025年Q3季度", "revenue": "2716.0亿元", "yoy_growth": "-17.9%", "net_incom...`
  - Prediction: `在2025年第三季度，海天味业的总营收达到2716.0亿元人民币，较去年同期下滑了17.9%，显示出市场环境对消费板块的影响。尽管整体营收有所减少，但公司仍实现了2835.7亿元的净利润，毛利率为19...`
  - Reference: `海天味业2025年Q3季度实现营业收入2716.0亿元,同比下滑17.9%。净利润达到2835.7亿元,利润率为19.7%。消费板块收入下降17.2%,面临一定经营压力。展望未来,成本优化措施有望帮助...`


### TUNED

- Error rate: 45.0%
- Total errors: 18

**Error Distribution:**

- partial_match: 13
- hallucination: 5

**Example Failures:**

*hallucination* (task: fin_summary):
  - Input: `万科A发布了2025年Q3季度业绩报告。报告期内,公司实现营业收入1479.7亿元,同比下降34.3%。归属于上市公司股东的净利润为3292.9亿元,营业收入为10.7%。公司核心业务保持稳健增长,但...`
  - Prediction: `{"summary": "万科A2025年Q3季度实现营收1479.7亿元,同比下滑34.3%。净利润为3292.9亿元。公司核心业务表现稳健。", "key_points": ["营业收入1479....`
  - Reference: `{"summary": "万科A2025年Q3季度实现营收1479.7亿元,同比下降34.3%。净利润为3292.9亿元。公司核心业务表现稳健。", "key_points": ["营业收入1479....`

*partial_match* (task: analysis_gen):
  - Input: `{"company": "海天味业", "period": "2025年Q3季度", "revenue": "2716.0亿元", "yoy_growth": "-17.9%", "net_incom...`
  - Prediction: `海天味业2025年Q3季度实现营业收入2716.0亿元,同比下降17.9%。净利润达到2835.7亿元,利润率为19.7%。消费板块收入下降17.2%,显示出强劲的增长动力。展望未来,持续的成本优化措...`
  - Reference: `海天味业2025年Q3季度实现营业收入2716.0亿元,同比下滑17.9%。净利润达到2835.7亿元,利润率为19.7%。消费板块收入下降17.2%,面临一定经营压力。展望未来,成本优化措施有望帮助...`

