"""Day 17 LLM planner prompt builder (§17.4).

Two things are deliberately *excluded* from the system prompt, both measured
costs from the Day 17 plan:

- `FinancialQueryPlan.model_json_schema()` (§1.4: 963 tokens, 23.5% of the
  4096-token context window) — replaced with `_OPERATION_GUIDE`, a compact
  hand-written restatement of the arity table already enforced by
  `plan_validator.py`.
- Row labels of candidate tables (ADR 0006 decision B1) and the full 56-entry
  canonical-metric enum — both would add hundreds to over a thousand tokens
  (§1.4) for a case `MetricSelector` already handles without them: every
  metric field accepts a `raw_text` fallback, exactly like
  `rule_planner._metric_selector` already falls back to the question's own
  surface text when it cannot resolve a canonical name.

Few-shots are drawn from real questions wherever one exists (gold70 or the
Day 10 entity-case corpus); operations with zero observed occurrences
(`ratio`, `average`, `sum`, `rank` — Day 16 §1.8) are demonstrated with a
synthetic question paired with a real fact from the release, following the
same precedent as the Day 15 golden JSON examples
(`tests/golden/plans/valid/*.json`). Each example's `provenance` records
which case it is, so nobody mistakes a synthetic question for measured data.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass


@dataclass(frozen=True)
class FewShotExample:
    question: str
    plan: dict[str, object]
    provenance: str


_OPERATION_GUIDE = textwrap.dedent(
    """
    Bạn là bộ lập kế hoạch truy vấn báo cáo tài chính. Đọc câu hỏi tiếng Việt và
    trả về DUY NHẤT một object JSON theo đúng một trong các operation sau (chọn
    dựa vào số company/period/metric mà câu hỏi nhắc tới):

    - "lookup": 1 company, 1 period, "metric".
    - "compare": 1 company, 1 period, "metric_a" + "metric_b" (hai đại lượng cùng kỳ).
    - "compare_companies": >=2 company, 1 period, "metric" (đặt cạnh nhau, không xếp hạng).
    - "difference": 1 company, 2 period (thứ tự tăng dần), "metric".
    - "growth_rate": 1 company, 2 period (thứ tự tăng dần), "metric", "expected_unit": "percent".
    - "ratio": 1 company, 1 period, "numerator_metric" + "denominator_metric".
    - "average" / "sum": đúng một trong companies/periods có nhiều hơn 1 phần tử,
      phần còn lại có đúng 1 phần tử, "metric".
    - "rank": >=2 company, 1 period, "metric", "top_k" (1 <= top_k < số company).

    Mỗi trường metric (metric / metric_a / metric_b / numerator_metric /
    denominator_metric) là một trong hai dạng:
    - {"canonical": "<tên_canonical>"} nếu bạn chắc chắn khớp một chỉ tiêu tài
      chính tiêu chuẩn (ví dụ: net_revenue, total_assets, profit_after_tax).
    - {"raw_text": "<nguyên văn cụm từ trong câu hỏi>"} nếu không chắc chắn.

    KHÔNG bao gồm trường "candidate_table_ids" — hệ thống sẽ tự điền. KHÔNG
    thêm giải thích, chỉ trả về JSON.
    """
).strip()

_FEW_SHOTS: tuple[FewShotExample, ...] = (
    FewShotExample(
        question="Tra cứu doanh thu thuần của NVL năm 2023.",
        plan={
            "operation": "lookup",
            "companies": ["NVL"],
            "periods": ["2023"],
            "metric": {"canonical": "net_revenue"},
        },
        provenance="gold70 retq_0a32a6d94a6e7bad8479d11ebbc10495710bc76f86ee2b0bde7d77462fa29d99",
    ),
    FewShotExample(
        question="Tra cứu cho vay khách hàng và chứng khoán đầu tư của STB tại cuối năm 2024.",
        plan={
            "operation": "compare",
            "companies": ["STB"],
            "periods": ["2024"],
            "metric_a": {"canonical": "loans_to_customers"},
            "metric_b": {"canonical": "investment_securities"},
        },
        provenance=(
            "gold70 real question (Day 17 plan §1.1 multi_metric_unsupported shape "
            "(1 company, 1 period, 2 metrics) that the rule planner does not route "
            "to `compare` yet)"
        ),
    ),
    FewShotExample(
        question="So sánh Các khoản giảm trừ doanh thu giữa GEG và GEX năm 2017.",
        plan={
            "operation": "compare_companies",
            "companies": ["GEG", "GEX"],
            "periods": ["2017"],
            "metric": {"canonical": "sales_deductions"},
        },
        provenance="entity-cases-v1 template two_companies (ADR 0005)",
    ),
    FewShotExample(
        question="So sánh doanh thu thuần của VGT giữa năm 2022 và năm 2023.",
        plan={
            "operation": "difference",
            "companies": ["VGT"],
            "periods": ["2022", "2023"],
            "metric": {"canonical": "net_revenue"},
        },
        provenance="gold70, real question",
    ),
    FewShotExample(
        question="Tính tốc độ tăng trưởng doanh thu thuần của NVL từ năm 2022 đến năm 2023.",
        plan={
            "operation": "growth_rate",
            "companies": ["NVL"],
            "periods": ["2022", "2023"],
            "metric": {"canonical": "net_revenue"},
            "expected_unit": "percent",
        },
        provenance="gold70 retq_5a293140dab6370835bb93b17bb0503467626e4386d5e5ca5264afd3d2cff41b",
    ),
    FewShotExample(
        question="Tỷ lệ doanh thu thuần trên tổng tài sản của PNJ năm 2015 là bao nhiêu?",
        plan={
            "operation": "ratio",
            "companies": ["PNJ"],
            "periods": ["2015"],
            "numerator_metric": {"canonical": "net_revenue"},
            "denominator_metric": {"canonical": "total_assets"},
            "expected_unit": "ratio",
        },
        provenance=(
            "synthetic question, real fact from tests/golden/plans/valid/ratio.json "
            "(Day 16 §1.8: 0 real 'ratio'-shaped questions observed)"
        ),
    ),
    FewShotExample(
        question="Tổng tài sản bình quân của VIB giai đoạn 2021-2023 là bao nhiêu?",
        plan={
            "operation": "average",
            "companies": ["VIB"],
            "periods": ["2021", "2022", "2023"],
            "metric": {"canonical": "total_assets"},
        },
        provenance=(
            "synthetic question, real fact from tests/golden/plans/valid/average.json "
            "(Day 16 §1.8: 0 real 'average'-shaped questions observed)"
        ),
    ),
    FewShotExample(
        question="Tổng doanh thu thuần năm 2021 của GEX và PNJ cộng lại là bao nhiêu?",
        plan={
            "operation": "sum",
            "companies": ["GEX", "PNJ"],
            "periods": ["2021"],
            "metric": {"canonical": "net_revenue"},
        },
        provenance=(
            "synthetic question, real fact from tests/golden/plans/valid/sum.json "
            "(Day 16 §1.8: 0 real 'sum'-shaped questions observed)"
        ),
    ),
    FewShotExample(
        question="Xếp hạng 2 công ty có doanh thu thuần cao nhất năm 2021 giữa GEX, PNJ và KHG.",
        plan={
            "operation": "rank",
            "companies": ["GEX", "PNJ", "KHG"],
            "periods": ["2021"],
            "metric": {"canonical": "net_revenue"},
            "top_k": 2,
        },
        provenance=(
            "synthetic question, real fact from tests/golden/plans/valid/rank.json "
            "(Day 16 §1.8: 0 real 'rank'-shaped questions observed)"
        ),
    ),
)


def _render_few_shots() -> str:
    blocks = []
    for example in _FEW_SHOTS:
        rendered_plan = json.dumps(example.plan, ensure_ascii=False, separators=(",", ":"))
        blocks.append(f"Câu hỏi: {example.question}\nJSON: {rendered_plan}")
    return "\n\n".join(blocks)


def build_system_prompt() -> str:
    """Return the fixed, deterministic system prompt (cacheable across calls)."""
    return f"{_OPERATION_GUIDE}\n\nVí dụ:\n\n{_render_few_shots()}"


def build_user_prompt(question: str) -> str:
    """Return the per-question user prompt."""
    return f"Câu hỏi: {question}\nJSON:"
