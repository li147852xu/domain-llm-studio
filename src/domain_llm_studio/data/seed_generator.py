"""Bilingual seed data generator using template-based synthesis.

Generates realistic financial/enterprise domain examples for all four tasks
using Jinja2 templates with randomized entity/value slots and bilingual
vocabulary banks (Chinese + English).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# Domain Vocabulary Banks
# ---------------------------------------------------------------------------

COMPANIES_EN = [
    "Alphabet Inc.", "Amazon.com Inc.", "Apple Inc.", "Microsoft Corp.",
    "Meta Platforms", "Tesla Inc.", "NVIDIA Corp.", "JPMorgan Chase",
    "Goldman Sachs", "Bank of America", "Berkshire Hathaway", "Visa Inc.",
    "Mastercard Inc.", "Netflix Inc.", "Salesforce Inc.", "Adobe Inc.",
    "Intel Corp.", "Qualcomm Inc.", "Cisco Systems", "Oracle Corp.",
]

COMPANIES_ZH = [
    "贵州茅台", "中国平安", "招商银行", "宁德时代", "比亚迪",
    "腾讯控股", "阿里巴巴", "京东集团", "美团", "字节跳动",
    "中信证券", "工商银行", "建设银行", "中国移动", "海天味业",
    "恒瑞医药", "迈瑞医疗", "隆基绿能", "万科A", "中国中免",
]

EVENT_TYPES = [
    "earnings", "acquisition", "partnership", "layoff",
    "product_launch", "regulatory", "funding", "leadership_change",
]

EVENT_TYPES_ZH = {
    "earnings": "财报发布",
    "acquisition": "收购",
    "partnership": "战略合作",
    "layoff": "裁员",
    "product_launch": "产品发布",
    "regulatory": "监管事件",
    "funding": "融资",
    "leadership_change": "高管变动",
}

METRICS_EN = [
    "revenue", "net_income", "operating_profit", "gross_margin",
    "EPS", "free_cash_flow", "total_assets", "debt_ratio",
    "R&D_expense", "customer_count", "market_share", "headcount",
]

METRICS_ZH = [
    "营业收入", "净利润", "营业利润", "毛利率",
    "每股收益", "自由现金流", "总资产", "资产负债率",
    "研发费用", "客户数量", "市场份额", "员工人数",
]

QUARTERS = ["Q1", "Q2", "Q3", "Q4"]
YEARS = ["2023", "2024", "2025"]
DIRECTIONS = ["increase", "decrease"]
SENTIMENTS = ["positive", "negative", "neutral"]

RISK_FACTORS_EN = [
    "increased competition in core markets",
    "rising raw material costs",
    "potential regulatory tightening",
    "foreign exchange headwinds",
    "slowing consumer demand",
    "supply chain disruptions",
    "cybersecurity threats",
    "talent retention challenges",
]

RISK_FACTORS_ZH = [
    "核心市场竞争加剧",
    "原材料成本上涨",
    "潜在监管收紧",
    "汇率波动风险",
    "消费需求放缓",
    "供应链中断风险",
    "网络安全威胁",
    "人才流失风险",
]

OPPORTUNITY_FACTORS_EN = [
    "expansion into emerging markets",
    "growing demand for AI solutions",
    "strategic partnerships pipeline",
    "margin improvement from cost optimization",
    "new product launch cycle",
    "favorable regulatory changes",
]

OPPORTUNITY_FACTORS_ZH = [
    "新兴市场拓展机会",
    "AI解决方案需求增长",
    "战略合作伙伴关系拓展",
    "成本优化带来的利润提升",
    "新产品发布周期",
    "有利的政策环境变化",
]


def _rand_pct(rng: random.Random) -> str:
    return f"{rng.uniform(1.5, 35.0):.1f}%"


def _rand_amount_en(rng: random.Random) -> str:
    val = rng.uniform(0.5, 150.0)
    return f"${val:.1f} billion" if val > 10 else f"${val:.2f} billion"


def _rand_amount_zh(rng: random.Random) -> str:
    val = rng.uniform(5.0, 5000.0)
    return f"{val:.1f}亿元"


def _rand_date(rng: random.Random) -> str:
    y = rng.choice(YEARS)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y}-{m:02d}-{d:02d}"


# ---------------------------------------------------------------------------
# Task 1: Financial Document Summarization
# ---------------------------------------------------------------------------

def _gen_summary_en(rng: random.Random) -> dict:
    company = rng.choice(COMPANIES_EN)
    quarter = rng.choice(QUARTERS)
    year = rng.choice(YEARS)
    rev = _rand_amount_en(rng)
    growth = _rand_pct(rng)
    net = _rand_amount_en(rng)
    margin = _rand_pct(rng)
    metric = rng.choice(METRICS_EN[:4])
    direction = rng.choice(DIRECTIONS)
    risks = rng.sample(RISK_FACTORS_EN, k=rng.randint(1, 3))
    opps = rng.sample(OPPORTUNITY_FACTORS_EN, k=rng.randint(1, 2))

    document = (
        f"{company} reported {quarter} {year} financial results. "
        f"Total revenue reached {rev}, representing a {growth} "
        f"{'increase' if direction == 'increase' else 'decrease'} year-over-year. "
        f"Net income came in at {net}, with {metric} at {margin}. "
        f"The company highlighted strong performance in its core business segments, "
        f"while noting headwinds from {rng.choice(RISK_FACTORS_EN)}. "
        f"Management raised full-year guidance citing {rng.choice(OPPORTUNITY_FACTORS_EN)}. "
        f"Operating expenses were well-controlled, with R&D investment increasing to "
        f"support long-term growth initiatives. The board approved a quarterly dividend "
        f"of {rng.uniform(0.1, 2.5):.2f} per share."
    )

    key_points = [
        f"Revenue of {rev} ({growth} YoY {'growth' if direction == 'increase' else 'decline'})",
        f"Net income at {net}",
        f"{metric} at {margin}",
        "Full-year guidance raised",
    ]

    output = {
        "summary": (
            f"{company} delivered {quarter} {year} results with revenue of {rev}, "
            f"a {growth} YoY {'increase' if direction == 'increase' else 'decrease'}. "
            f"Net income reached {net}. Management raised guidance for the full year."
        ),
        "key_points": key_points,
        "risks": risks,
        "opportunities": opps,
    }

    return {
        "task": "fin_summary",
        "lang": "en",
        "input": document,
        "output": json.dumps(output, ensure_ascii=False),
    }


def _gen_summary_zh(rng: random.Random) -> dict:
    company = rng.choice(COMPANIES_ZH)
    quarter = rng.choice(QUARTERS)
    year = rng.choice(YEARS)
    rev = _rand_amount_zh(rng)
    growth = _rand_pct(rng)
    net = _rand_amount_zh(rng)
    margin = _rand_pct(rng)
    metric = rng.choice(METRICS_ZH[:4])
    direction = rng.choice(DIRECTIONS)
    risks = rng.sample(RISK_FACTORS_ZH, k=rng.randint(1, 3))
    opps = rng.sample(OPPORTUNITY_FACTORS_ZH, k=rng.randint(1, 2))

    document = (
        f"{company}发布了{year}年{quarter}季度业绩报告。"
        f"报告期内，公司实现营业收入{rev}，同比{'增长' if direction == 'increase' else '下降'}{growth}。"
        f"归属于上市公司股东的净利润为{net}，{metric}为{margin}。"
        f"公司核心业务保持稳健增长，但也面临{rng.choice(RISK_FACTORS_ZH)}等挑战。"
        f"管理层表示将持续加大研发投入，推动{rng.choice(OPPORTUNITY_FACTORS_ZH)}。"
        f"期间费用率有所优化，经营现金流保持健康水平。"
    )

    key_points = [
        f"营业收入{rev}，同比{'增长' if direction == 'increase' else '下降'}{growth}",
        f"净利润{net}",
        f"{metric}为{margin}",
        "持续加大研发投入",
    ]

    output = {
        "summary": (
            f"{company}{year}年{quarter}季度实现营收{rev}，"
            f"同比{'增长' if direction == 'increase' else '下降'}{growth}。"
            f"净利润为{net}。公司核心业务表现稳健。"
        ),
        "key_points": key_points,
        "risks": risks,
        "opportunities": opps,
    }

    return {
        "task": "fin_summary",
        "lang": "zh",
        "input": document,
        "output": json.dumps(output, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Task 2: Event & Entity Extraction
# ---------------------------------------------------------------------------

_EVENT_TEMPLATES_EN = [
    "{company} announced on {date} that its {quarter} {year} {metric} reached {amount}, a {pct} {direction} year-over-year. Analysts viewed the results as {sentiment}.",
    "{company} completed the acquisition of {target} for {amount} on {date}. The deal is expected to strengthen {company}'s position in {sector}.",
    "{company} announced a strategic partnership with {partner} on {date} to jointly develop {product}. The collaboration is viewed as {sentiment} for both parties.",
    "{company} confirmed plans to reduce its workforce by {headcount} employees as part of a restructuring plan announced on {date}.",
    "{company} launched {product} on {date}, marking the company's entry into the {sector} market.",
    "{company} received regulatory approval for {product} from {regulator} on {date}.",
]

_EVENT_TEMPLATES_ZH = [
    "{company}于{date}公告，{quarter}{year}年度{metric}达到{amount}，同比{direction}{pct}。市场对此反应{sentiment}。",
    "{company}于{date}宣布完成对{target}的收购，交易金额为{amount}，预计将增强公司在{sector}领域的竞争力。",
    "{company}与{partner}于{date}签署战略合作协议，双方将在{product}领域开展深度合作。",
    "{company}于{date}宣布裁员{headcount}人的重组计划，以优化公司运营效率。",
    "{company}于{date}正式发布{product}，标志着公司进入{sector}市场。",
    "{company}的{product}于{date}获得{regulator}的监管批准。",
]

SECTORS_EN = ["cloud computing", "electric vehicles", "fintech", "healthcare", "semiconductors"]
SECTORS_ZH = ["云计算", "新能源汽车", "金融科技", "医疗健康", "半导体"]
PRODUCTS_EN = ["AI assistant platform", "next-gen chip", "cloud service suite", "payment system"]
PRODUCTS_ZH = ["AI助手平台", "新一代芯片", "云服务套件", "支付系统"]
REGULATORS_EN = ["the SEC", "the FDA", "the FTC", "EU regulators"]
REGULATORS_ZH = ["证监会", "药监局", "银保监会", "工信部"]


_TEMPLATE_EVENT_TYPES = ["earnings", "acquisition", "partnership", "layoff", "product_launch", "regulatory"]


def _gen_extraction_en(rng: random.Random) -> dict:
    company = rng.choice(COMPANIES_EN)
    template_idx = rng.randrange(len(_EVENT_TEMPLATES_EN))
    event_type = _TEMPLATE_EVENT_TYPES[template_idx]
    template = _EVENT_TEMPLATES_EN[template_idx]
    date = _rand_date(rng)
    quarter = rng.choice(QUARTERS)
    year = rng.choice(YEARS)
    metric = rng.choice(METRICS_EN)
    amount = _rand_amount_en(rng)
    pct = _rand_pct(rng)
    direction = rng.choice(DIRECTIONS)
    sentiment = rng.choice(SENTIMENTS)
    target = rng.choice([c for c in COMPANIES_EN if c != company])
    partner = rng.choice([c for c in COMPANIES_EN if c != company])

    text = template.format(
        company=company, date=date, quarter=quarter, year=year,
        metric=metric, amount=amount, pct=pct,
        direction=direction, sentiment=sentiment,
        target=target, partner=partner,
        product=rng.choice(PRODUCTS_EN), sector=rng.choice(SECTORS_EN),
        headcount=rng.randint(500, 15000), regulator=rng.choice(REGULATORS_EN),
    )

    events = [{
        "company": company,
        "event_type": event_type,
        "date": date,
        "metric": metric if event_type == "earnings" else None,
        "change_direction": direction if event_type == "earnings" else None,
        "sentiment": sentiment,
    }]

    return {
        "task": "event_extraction",
        "lang": "en",
        "input": text,
        "output": json.dumps(events, ensure_ascii=False),
    }


def _gen_extraction_zh(rng: random.Random) -> dict:
    company = rng.choice(COMPANIES_ZH)
    template_idx = rng.randrange(len(_EVENT_TEMPLATES_ZH))
    event_type = _TEMPLATE_EVENT_TYPES[template_idx]
    template = _EVENT_TEMPLATES_ZH[template_idx]
    date = _rand_date(rng)
    quarter = rng.choice(QUARTERS)
    year = rng.choice(YEARS)
    metric = rng.choice(METRICS_ZH)
    amount = _rand_amount_zh(rng)
    pct = _rand_pct(rng)
    direction_str = "增长" if rng.choice(DIRECTIONS) == "increase" else "下降"
    direction = "increase" if direction_str == "增长" else "decrease"
    sentiment_map = {"positive": "积极", "negative": "消极", "neutral": "中性"}
    sentiment = rng.choice(SENTIMENTS)
    target = rng.choice([c for c in COMPANIES_ZH if c != company])
    partner = rng.choice([c for c in COMPANIES_ZH if c != company])

    text = template.format(
        company=company, date=date, quarter=quarter, year=year,
        metric=metric, amount=amount, pct=pct,
        direction=direction_str, sentiment=sentiment_map[sentiment],
        target=target, partner=partner,
        product=rng.choice(PRODUCTS_ZH), sector=rng.choice(SECTORS_ZH),
        headcount=rng.randint(500, 15000), regulator=rng.choice(REGULATORS_ZH),
    )

    events = [{
        "company": company,
        "event_type": event_type,
        "date": date,
        "metric": metric if event_type == "earnings" else None,
        "change_direction": direction if event_type == "earnings" else None,
        "sentiment": sentiment,
    }]

    return {
        "task": "event_extraction",
        "lang": "zh",
        "input": text,
        "output": json.dumps(events, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Task 3: Document QA
# ---------------------------------------------------------------------------

_QA_CONTEXTS_EN = [
    (
        "{company} reported {quarter} {year} revenue of {rev}, a {pct} increase from the prior year. "
        "The growth was primarily driven by {driver}. Operating expenses rose {opex_pct} due to "
        "increased investment in R&D. The company ended the quarter with {cash} in cash and equivalents.",
        [
            ("What was {company}'s revenue in {quarter} {year}?", "{rev}"),
            ("What drove the revenue growth?", "{driver}"),
            ("How much cash did {company} have?", "{cash}"),
        ],
    ),
    (
        "On {date}, {company} announced the appointment of {person} as the new CEO, replacing {old_person} "
        "who served in the role for {tenure} years. The board cited the need for fresh leadership "
        "to navigate the company's expansion into {sector}.",
        [
            ("Who is the new CEO of {company}?", "{person}"),
            ("How long did {old_person} serve as CEO?", "{tenure} years"),
            ("Why was the leadership change made?", "to navigate the company's expansion into {sector}"),
        ],
    ),
]

_QA_CONTEXTS_ZH = [
    (
        "{company}发布{year}年{quarter}季度业绩报告，实现营业收入{rev}，同比增长{pct}。"
        "增长主要来自于{driver}。公司研发费用同比增长{opex_pct}，持续加大技术投入。"
        "期末现金及现金等价物为{cash}。",
        [
            ("{company}{quarter}季度的营业收入是多少？", "{rev}"),
            ("收入增长的主要驱动因素是什么？", "{driver}"),
            ("{company}期末现金储备是多少？", "{cash}"),
        ],
    ),
    (
        "{company}于{date}公告，任命{person}为公司新任首席执行官，接替任职{tenure}年的{old_person}。"
        "董事会表示，此次管理层调整旨在推动公司在{sector}领域的战略转型。",
        [
            ("{company}的新任CEO是谁？", "{person}"),
            ("{old_person}担任CEO多长时间？", "{tenure}年"),
            ("为什么进行管理层调整？", "推动公司在{sector}领域的战略转型"),
        ],
    ),
]

PERSONS_EN = ["Sarah Johnson", "Michael Chen", "David Park", "Lisa Wang", "James Miller"]
PERSONS_ZH = ["张伟", "李明", "王芳", "刘强", "陈晓华"]
DRIVERS_EN = [
    "strong cloud computing demand", "growing subscription revenue",
    "robust consumer spending", "international market expansion",
    "enterprise software adoption",
]
DRIVERS_ZH = [
    "云计算业务强劲增长", "订阅收入持续提升",
    "消费需求旺盛", "海外市场拓展加速",
    "企业级软件需求增加",
]


def _gen_qa_en(rng: random.Random) -> dict:
    company = rng.choice(COMPANIES_EN)
    template_ctx, qa_pairs = rng.choice(_QA_CONTEXTS_EN)
    params = {
        "company": company,
        "quarter": rng.choice(QUARTERS),
        "year": rng.choice(YEARS),
        "rev": _rand_amount_en(rng),
        "pct": _rand_pct(rng),
        "driver": rng.choice(DRIVERS_EN),
        "opex_pct": _rand_pct(rng),
        "cash": _rand_amount_en(rng),
        "date": _rand_date(rng),
        "person": rng.choice(PERSONS_EN),
        "old_person": rng.choice(PERSONS_EN),
        "tenure": str(rng.randint(3, 15)),
        "sector": rng.choice(SECTORS_EN),
    }
    context = template_ctx.format(**params)
    q_template, a_template = rng.choice(qa_pairs)
    question = q_template.format(**params)
    answer = a_template.format(**params)

    output = {
        "answer": answer,
        "evidence_span": answer if len(answer) < 100 else answer[:100],
    }

    return {
        "task": "doc_qa",
        "lang": "en",
        "input": json.dumps({"context": context, "question": question}, ensure_ascii=False),
        "output": json.dumps(output, ensure_ascii=False),
    }


def _gen_qa_zh(rng: random.Random) -> dict:
    company = rng.choice(COMPANIES_ZH)
    template_ctx, qa_pairs = rng.choice(_QA_CONTEXTS_ZH)
    params = {
        "company": company,
        "quarter": rng.choice(QUARTERS),
        "year": rng.choice(YEARS),
        "rev": _rand_amount_zh(rng),
        "pct": _rand_pct(rng),
        "driver": rng.choice(DRIVERS_ZH),
        "opex_pct": _rand_pct(rng),
        "cash": _rand_amount_zh(rng),
        "date": _rand_date(rng),
        "person": rng.choice(PERSONS_ZH),
        "old_person": rng.choice(PERSONS_ZH),
        "tenure": str(rng.randint(3, 15)),
        "sector": rng.choice(SECTORS_ZH),
    }
    context = template_ctx.format(**params)
    q_template, a_template = rng.choice(qa_pairs)
    question = q_template.format(**params)
    answer = a_template.format(**params)

    output = {
        "answer": answer,
        "evidence_span": answer if len(answer) < 100 else answer[:100],
    }

    return {
        "task": "doc_qa",
        "lang": "zh",
        "input": json.dumps({"context": context, "question": question}, ensure_ascii=False),
        "output": json.dumps(output, ensure_ascii=False),
    }


# ---------------------------------------------------------------------------
# Task 4: Structured Analysis Generation
# ---------------------------------------------------------------------------

def _gen_analysis_en(rng: random.Random) -> dict:
    company = rng.choice(COMPANIES_EN)
    period = f"{rng.choice(QUARTERS)} {rng.choice(YEARS)}"
    rev = _rand_amount_en(rng)
    growth = _rand_pct(rng)
    net = _rand_amount_en(rng)
    margin = _rand_pct(rng)
    segment = rng.choice([
        "Cloud revenue grew " + _rand_pct(rng),
        "Consumer segment declined " + _rand_pct(rng),
        "Enterprise division expanded by " + _rand_pct(rng),
        "International sales increased " + _rand_pct(rng),
    ])

    structured_data = {
        "company": company,
        "period": period,
        "revenue": rev,
        "yoy_growth": f"+{growth}" if rng.random() > 0.3 else f"-{growth}",
        "net_income": net,
        "margin": margin,
        "segment_highlight": segment,
    }

    growth_dir = "growth" if structured_data["yoy_growth"].startswith("+") else "decline"
    memo = (
        f"{company} reported {period} revenue of {rev}, "
        f"representing a {growth} year-over-year {growth_dir}. "
        f"Net income reached {net} with margins at {margin}. "
        f"{segment}, indicating {'strong momentum' if growth_dir == 'growth' else 'headwinds'} "
        f"in this business line. "
        f"Looking ahead, {'continued investment in growth initiatives' if growth_dir == 'growth' else 'cost optimization measures'} "
        f"should {'support further expansion' if growth_dir == 'growth' else 'help stabilize margins'} "
        f"in the coming quarters."
    )

    return {
        "task": "analysis_gen",
        "lang": "en",
        "input": json.dumps(structured_data, ensure_ascii=False),
        "output": memo,
    }


def _gen_analysis_zh(rng: random.Random) -> dict:
    company = rng.choice(COMPANIES_ZH)
    period = f"{rng.choice(YEARS)}年{rng.choice(QUARTERS)}季度"
    rev = _rand_amount_zh(rng)
    growth = _rand_pct(rng)
    net = _rand_amount_zh(rng)
    margin = _rand_pct(rng)
    segment = rng.choice([
        "云业务收入增长" + _rand_pct(rng),
        "消费板块收入下降" + _rand_pct(rng),
        "企业服务部门扩张" + _rand_pct(rng),
        "海外销售增长" + _rand_pct(rng),
    ])

    growth_sign = rng.random() > 0.3
    structured_data = {
        "company": company,
        "period": period,
        "revenue": rev,
        "yoy_growth": f"+{growth}" if growth_sign else f"-{growth}",
        "net_income": net,
        "margin": margin,
        "segment_highlight": segment,
    }

    growth_dir = "增长" if growth_sign else "下滑"
    memo = (
        f"{company}{period}实现营业收入{rev}，同比{growth_dir}{growth}。"
        f"净利润达到{net}，利润率为{margin}。"
        f"{segment}，{'显示出强劲的增长动力' if growth_sign else '面临一定经营压力'}。"
        f"展望未来，{'持续的增长投入' if growth_sign else '成本优化措施'}"
        f"有望{'推动业绩进一步提升' if growth_sign else '帮助稳定利润水平'}。"
    )

    return {
        "task": "analysis_gen",
        "lang": "zh",
        "input": json.dumps(structured_data, ensure_ascii=False),
        "output": memo,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

GENERATORS = {
    "fin_summary": (_gen_summary_en, _gen_summary_zh),
    "event_extraction": (_gen_extraction_en, _gen_extraction_zh),
    "doc_qa": (_gen_qa_en, _gen_qa_zh),
    "analysis_gen": (_gen_analysis_en, _gen_analysis_zh),
}


def generate_seed_data(
    output_dir: Path,
    num_samples_per_task: int = 50,
    seed: int = 42,
    languages: list[str] | None = None,
) -> dict[str, int]:
    """Generate seed data for all tasks. Returns count per task."""
    languages = languages or ["en", "zh"]
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    for task_name, (gen_en, gen_zh) in GENERATORS.items():
        samples = []
        for _ in range(num_samples_per_task):
            if "en" in languages:
                samples.append(gen_en(rng))
            if "zh" in languages:
                samples.append(gen_zh(rng))

        out_file = output_dir / f"{task_name}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in samples:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        counts[task_name] = len(samples)

    return counts
