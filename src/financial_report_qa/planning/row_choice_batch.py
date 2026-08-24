"""Dựng payload batch cho bước sinh chương trình masked (spec 2026-08-24 §4.3).

LLM nhận câu hỏi + công ty/kỳ đã tách + danh sách ô ứng viên đã đánh số,
trả về một chương trình với các vị trí ô (`cells`) -- nó không bao giờ thấy
giá trị số của ô, và không bao giờ trả về giá trị -- đáp án luôn được tính
lại từ CSV bằng pandas trong sandbox.

Thứ tự phần tử trong `candidates` **là** hợp đồng: `ProgramDecision.cells`
map ngược về ứng viên bằng chính vị trí này.
"""

from __future__ import annotations

from collections.abc import Sequence

from financial_report_qa.execution.program_contracts import CellCandidate
from financial_report_qa.planning.entity_contracts import QueryEntities


def _cell_candidate_payload(candidate: CellCandidate) -> dict[str, object]:
    return {
        "index": candidate.index,
        "company_code": candidate.company_code,
        "row_path": candidate.row_path,
        "col_path": candidate.col_path,
        "period": candidate.period,
        "statement_type": candidate.statement_type,
        "unit": candidate.unit,
    }


def build_program_batch_payload(
    question_id: int,
    question: str,
    entities: QueryEntities,
    candidates: Sequence[CellCandidate],
) -> dict[str, object]:
    """Một dòng JSONL cho bước sinh chương trình masked (spec 2026-08-24 §4.3).

    Không trường nào mang giá trị ô hay điểm fusion (N7). Thứ tự `candidates`
    **là** hợp đồng: `ProgramDecision.cells` là vị trí trong chính danh sách
    này, nên sắp lại ở đây là làm sai mọi quyết định đã sinh.
    """
    return {
        "question_id": question_id,
        "question": question,
        "companies": list(entities.company_codes),
        "periods": list(entities.periods),
        "candidates": [_cell_candidate_payload(candidate) for candidate in candidates],
    }
