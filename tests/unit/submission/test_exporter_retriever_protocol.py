"""Runtime proof that both concrete table retrievers satisfy the protocol
the submission exporter (and `sweep-k`) depend on. If either drifts from the
`retrieve` signature, this fails here instead of three hours into an export."""

from financial_report_qa.retrieval.fusion import FusionService
from financial_report_qa.retrieval.live_query import TableRetriever
from financial_report_qa.retrieval.service import RetrievalService


def test_both_retrievers_satisfy_the_protocol_the_exporter_depends_on() -> None:
    # Nếu một trong hai lệch chữ ký, mypy bắt được ở đây trước khi chạy
    # export 3 tiếng mới phát hiện.
    assert issubclass(RetrievalService, TableRetriever)
    assert issubclass(FusionService, TableRetriever)


def test_protocol_is_checkable_on_instances_too() -> None:
    # The exporter receives instances, not classes: @runtime_checkable must
    # hold for isinstance as well, and TableRetriever declares methods only,
    # so issubclass/isinstance are both legal (no data member on the protocol
    # itself -- the ones inside the returned traces never participate).
    class _Fake:
        def retrieve(
            self,
            query: str,
            *,
            filters: object,
            k: int = 10,
            question_id: str | None = None,
        ) -> object:
            return None

    assert isinstance(_Fake(), TableRetriever)
