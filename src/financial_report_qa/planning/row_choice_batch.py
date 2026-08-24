"""Dựng payload batch cho bước LLM chọn dòng (thiết kế 2026-08-22 §5.3).

LLM nhận câu hỏi + công ty/kỳ đã tách + danh sách dòng ứng viên đã đánh số,
trả về operation và các cặp (company_code, row_index). Nó không bao giờ thấy
giá trị số của ô, và không bao giờ trả về giá trị -- đáp án luôn được tính
lại từ CSV bằng pandas trong sandbox. Đó là điều giữ cho compiler
deterministic và compliance linter còn ý nghĩa.

Thứ tự phần tử trong `candidates` **là** hợp đồng: phần quyết định dòng map
`chosen` ngược về ứng viên bằng chính vị trí này.
"""

from __future__ import annotations

from collections.abc import Sequence

from financial_report_qa.execution.program_contracts import CellCandidate
from financial_report_qa.planning.entity_contracts import QueryEntities
from financial_report_qa.retrieval.row_fusion_contracts import RowFusedCandidate


def _candidate_payload(index: int, candidate: RowFusedCandidate) -> dict[str, object]:
    metadata = candidate.metadata
    return {
        "index": index,
        "company_code": metadata.company_code,
        "row_label": metadata.row_label_raw or "",
        "row_group_context": metadata.row_group_context_raw,
        "statement_type": metadata.statement_type,
        "table_title": metadata.title,
        "periods": list(metadata.periods),
        "units": list(metadata.units),
    }


def build_batch_payload(
    question_id: int,
    question: str,
    entities: QueryEntities,
    candidates: Sequence[RowFusedCandidate],
) -> dict[str, object]:
    """Một dòng JSONL: câu hỏi + công ty/kỳ đã tách + ứng viên đánh số từ 0.

    `companies` và `periods` đi kèm để model biết câu hỏi có mấy công ty --
    đó là thứ quyết định nó phải trả về mấy chỉ số trong `chosen` và
    operation nào hợp lý (`rank`/`compare_companies` chỉ có nghĩa khi nhiều
    công ty).

    `candidates` phải đã ở đúng thứ tự retrieval-rank; thứ tự này **là** hợp
    đồng -- `question_plan.assemble_plan` map `chosen` ngược về ứng viên bằng
    chính vị trí này. Không sắp lại.

    Không trường nào mang giá trị ô hay điểm fusion (bất biến N7).
    """
    return {
        "question_id": question_id,
        "question": question,
        "companies": list(entities.company_codes),
        "periods": list(entities.periods),
        "candidates": [
            _candidate_payload(index, candidate) for index, candidate in enumerate(candidates)
        ],
    }


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
