"""Chọn k* cho nhánh retrieval bằng số đo, không bằng cảm tính.

`retrieval_scoring.sweep_k` đã cài đúng hai công thức của dashboard nhưng
chưa có caller: module này là caller đó. Nó chạy đúng pipeline live
(`retrieve_candidate_table_ids`, tức là đã gồm metadata filtering, fusion và
reranker nếu được truyền vào) trên tập gold, cắt danh sách đã xếp hạng ở
từng k, rồi báo cáo F2 macro và MRR5 macro song song.

`recommend_k` chọn F2 cao nhất và phá thế hoà bằng MRR5, KHÔNG tối đa hoá
một chỉ số đơn lẻ: F2 phạt precision khi k lớn, còn MRR5 gần như không đổi
khi k > 5, nên tối đa hoá riêng MRR5 sẽ đẩy k lên vô ích và mất điểm F2.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from financial_report_qa.retrieval.live_query import (
    TableRetriever,
    retrieve_candidate_table_ids,
)
from financial_report_qa.retrieval.rerank_contracts import DEFAULT_RERANK_DEPTH
from financial_report_qa.retrieval.reranker import Reranker
from financial_report_qa.retrieval.retrieval_scoring import sweep_k

DEFAULT_KS: tuple[int, ...] = (1, 2, 3, 5, 8, 10, 15)


class _GoldQuestion(Protocol):
    question: str
    gold_table_ids: tuple[str, ...]


@dataclass(frozen=True)
class SweepResult:
    """F2 macro và MRR5 macro tại một giá trị k."""

    k: int
    f2: float
    mrr5: float


def run_sweep(
    questions: Sequence[_GoldQuestion],
    service: TableRetriever,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    reranker: Reranker | None = None,
    rerank_depth: int = DEFAULT_RERANK_DEPTH,
) -> tuple[SweepResult, ...]:
    """Chạy pipeline một lần ở k lớn nhất, rồi cắt cho mọi k nhỏ hơn.

    Chạy đúng một lần cho mỗi câu (ở `max(ks)`) thay vì một lần cho mỗi
    (câu, k): danh sách trả về đã ở đúng thứ tự retrieval-rank nên cắt ngắn
    cho k nhỏ hơn cho kết quả y hệt như truy hồi lại với k đó, với chi phí
    bằng 1/len(ks).
    """
    if not ks:
        raise ValueError("ks must not be empty")
    depth = max(ks)
    effective_rerank_depth = max(rerank_depth, depth)

    ranked: dict[int, list[str]] = {}
    gold: dict[int, list[str]] = {}
    for index, question in enumerate(questions):
        ranked[index] = list(
            retrieve_candidate_table_ids(
                question.question,
                service,
                k=depth,
                reranker=reranker,
                rerank_depth=effective_rerank_depth,
            )
        )
        gold[index] = list(question.gold_table_ids)

    scored = sweep_k(ranked, gold, ks=tuple(ks))
    return tuple(SweepResult(k=k, f2=scored[k]["f2"], mrr5=scored[k]["mrr5"]) for k in ks)


def recommend_k(results: Sequence[SweepResult]) -> int:
    """F2 cao nhất; hoà thì MRR5 cao hơn thắng; vẫn hoà thì k nhỏ hơn thắng."""
    if not results:
        raise ValueError("cannot recommend k from an empty sweep")
    best = max(results, key=lambda item: (item.f2, item.mrr5, -item.k))
    return best.k


def render_sweep_markdown(results: Sequence[SweepResult], recommended_k: int) -> str:
    """Bảng Markdown một dòng một k, đánh dấu k* đã chọn."""
    lines = ["| k | F2 macro | MRR5 macro | |", "|---|---|---|---|"]
    for item in results:
        marker = " <-- k*" if item.k == recommended_k else ""
        lines.append(f"| {item.k} | {item.f2:.4f} | {item.mrr5:.4f} |{marker} |")
    return "\n".join(lines) + "\n"
