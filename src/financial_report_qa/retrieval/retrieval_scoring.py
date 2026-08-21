"""Đo truy hồi theo đúng công thức dashboard chấm điểm công khai của cuộc thi.

Dashboard có 10 cột: `EXECUTION ACCURACY, TABLES F2-MACRO, DOCS F2-MACRO,
TABLES PRECISION, TABLES RECALL, TABLES MRR5, DOCS PRECISION, DOCS RECALL,
DOCS MRR5, ANSWER ACCURACY`. Module này cài hai công thức chung cho cả nhánh
TABLES và nhánh DOCS (gọi hai lần với hai tập gold khác nhau):

    Precision = |đúng| / |đã truy hồi|
    Recall    = |đúng| / |liên quan|
    F2        = 5*P*R / (4*P + R)                    -- không phân biệt thứ tự
    MRR5      = 1 / rank(kết quả đúng đầu tiên trong top-5), 0 nếu không có

F2 nghiêng recall gấp 4 lần precision, nhưng precision giảm theo 1/k trong khi
recall bão hoà -- k lớn thường lợi cho recall, hại cho F2. MRR5 chỉ quan tâm
vị trí kết quả đúng ĐẦU TIÊN trong 5 phần tử đầu, nên gần như không đổi khi
k > 5. Hai chỉ số có thể đòi hỏi k khác nhau; `sweep_k` đo cả hai để người
dùng tự cân bằng, không chỉ tối ưu một chỉ số.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

_MRR_DEPTH = 5


def f2_score(predicted: Sequence[str], gold: Sequence[str]) -> float:
    """F2 cho một truy vấn. Không phân biệt thứ tự; trùng lặp trong
    `predicted` không được tính hai lần."""
    predicted_set = set(predicted)
    gold_set = set(gold)
    if not gold_set or not predicted_set:
        return 0.0
    hits = len(predicted_set & gold_set)
    if hits == 0:
        return 0.0
    precision = hits / len(predicted_set)
    recall = hits / len(gold_set)
    return 5 * precision * recall / (4 * precision + recall)


def mrr5_score(predicted_ranked: Sequence[str], gold: Sequence[str]) -> float:
    """MRR@5 cho một truy vấn: 1/hạng của kết quả đúng ĐẦU TIÊN trong 5 phần
    tử đầu của `predicted_ranked` (hạng 1-indexed). 0 nếu không có kết quả
    đúng nào trong top-5.

    Khác `f2_score`: thứ tự của `predicted_ranked` quyết định điểm số. Đầu
    vào PHẢI đã ở đúng thứ tự retrieval-rank (điểm cao nhất trước) -- một
    danh sách đúng tập nhưng sai thứ tự cho điểm sai mà không có cách nào
    phát hiện qua F2.
    """
    gold_set = set(gold)
    if not gold_set:
        return 0.0
    for rank, table_id in enumerate(predicted_ranked[:_MRR_DEPTH], start=1):
        if table_id in gold_set:
            return 1.0 / rank
    return 0.0


def macro_f2(
    predictions: Mapping[int, Sequence[str]], gold: Mapping[int, Sequence[str]]
) -> float:
    """Trung bình F2 trên mọi câu có gold. Câu thiếu dự đoán tính 0."""
    if not gold:
        return 0.0
    total = sum(f2_score(predictions.get(qid, ()), tables) for qid, tables in gold.items())
    return total / len(gold)


def macro_mrr5(
    predictions: Mapping[int, Sequence[str]], gold: Mapping[int, Sequence[str]]
) -> float:
    """Trung bình MRR@5 trên mọi câu có gold. Câu thiếu dự đoán tính 0."""
    if not gold:
        return 0.0
    total = sum(mrr5_score(predictions.get(qid, ()), tables) for qid, tables in gold.items())
    return total / len(gold)


def sweep_k(
    ranked: Mapping[int, Sequence[str]],
    gold: Mapping[int, Sequence[str]],
    ks: Sequence[int] = (1, 2, 3, 5, 8, 10, 15),
) -> dict[int, dict[str, float]]:
    """F2 và MRR5 macro khi cắt danh sách đã xếp hạng ở từng k.

    `ranked` phải đã ở đúng thứ tự retrieval-rank cho mỗi câu -- kết quả
    MRR5 vô nghĩa nếu không.
    """
    result: dict[int, dict[str, float]] = {}
    for k in ks:
        truncated = {qid: list(tables)[:k] for qid, tables in ranked.items()}
        result[k] = {
            "f2": macro_f2(truncated, gold),
            "mrr5": macro_mrr5(truncated, gold),
        }
    return result
