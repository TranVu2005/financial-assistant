"""Stable product error hierarchy."""


class FinancialReportQAError(RuntimeError):
    """Base class for expected domain and operational failures."""


class SourceIngestionError(FinancialReportQAError):
    """Base class for deterministic source-ingestion failures."""


class InvalidSourceDocumentError(SourceIngestionError):
    """The inventory record cannot be consumed by ingestion."""


class UnsupportedSourceEncodingError(SourceIngestionError):
    """The inventory-approved source encoding is unsupported or inconsistent."""


class SourceSnapshotMismatchError(SourceIngestionError):
    """The current source bytes differ from the immutable inventory record."""


class SourceReadError(SourceIngestionError):
    """The verified relative source path could not be read."""


class NormalizationError(FinancialReportQAError):
    """A normalization contract or ruleset is invalid."""


class DatasetBuildError(FinancialReportQAError):
    """A canonical dataset could not be built or verified."""


class DatasetPublicationError(DatasetBuildError):
    """A verified dataset release could not be safely published."""


class Week1GateError(FinancialReportQAError):
    """Base class for expected Week 1 gate workflow failures."""


class Week1GateInputError(Week1GateError):
    """Gate inputs, annotations, or release identity are invalid."""


class Week1GateSourceError(Week1GateError):
    """A source document cannot be re-verified against the manifest."""


class Week1GatePublicationError(Week1GateError):
    """Gate artifacts cannot be safely verified or published."""


class QualityGateError(FinancialReportQAError):
    """A quality gate threshold or invariant was violated."""


class RetrievalError(FinancialReportQAError):
    """Base class for deterministic retrieval failures."""


class RetrievalInputError(RetrievalError):
    """A release lock, gold record, or CLI input is invalid."""


class RetrievalArtifactError(RetrievalError):
    """A generated retrieval artifact is corrupt or mismatched."""


class RetrievalReleaseError(RetrievalInputError):
    """The Week 1 release lock cannot be used for retrieval."""


class RetrievalGoldError(RetrievalInputError):
    """Reviewed retrieval gold data violates its immutable contract."""


class DenseInputError(RetrievalInputError):
    """Dense retrieval inputs or encoder configuration are invalid."""


class DenseArtifactError(RetrievalArtifactError):
    """A dense corpus, index, cache, or evaluation artifact is invalid."""


class DenseModelError(RetrievalInputError):
    """A pinned dense encoder model cannot be loaded exactly."""


class FusionError(RetrievalError):
    """Base class for deterministic BM25/dense fusion failures."""


class FusionInputError(FusionError):
    """Fusion weights, gold, or CLI inputs are invalid."""


class FusionArtifactError(FusionError):
    """A fusion evaluation artifact is corrupt or mismatched."""


class RerankError(RetrievalError):
    """Cross-encoder reranking failed."""


class RerankInputError(RerankError):
    """Caller supplied invalid rerank input."""


class RerankModelError(RerankError):
    """Pinned reranker model is unavailable or produced an unusable score."""


class GraphError(RetrievalError):
    """Base class for deterministic GTR-lite graph failures."""


class GraphInputError(GraphError):
    """Graph build inputs, corpus, or CLI arguments are invalid."""


class GraphArtifactError(GraphError):
    """A generated graph artifact is corrupt or mismatched."""


class ExpansionError(RetrievalError):
    """Base class for deterministic Day 12 graph expansion failures."""


class ExpansionInputError(ExpansionError):
    """Expansion inputs, grid, or CLI arguments are invalid."""


class ExpansionArtifactError(ExpansionError):
    """An expansion evaluation artifact is corrupt or mismatched."""


class PlanningError(FinancialReportQAError):
    """Base class for deterministic query-planning failures."""


class PlanningInputError(PlanningError):
    """Entity-parser inputs, templates, or CLI arguments are invalid."""


class PlanningArtifactError(PlanningError):
    """A generated planning artifact is corrupt or mismatched."""


class TableFrameError(FinancialReportQAError):
    """A table cannot be reconstructed as a pandas DataFrame from a release."""


class LLMError(FinancialReportQAError):
    """Base class for Day 17 LLM-planner client failures."""


class LLMUnavailableError(LLMError):
    """The LLM endpoint could not be reached after bounded retries."""


class LLMServerError(LLMUnavailableError):
    """The endpoint answered, but with 5xx on every bounded retry.

    A subclass of `LLMUnavailableError` so existing handlers are unchanged:
    the distinction matters only where the caller can vary the request (see
    `llm_client.complete_json`'s unconstrained retry), because a server that
    replies at all is not offline.
    """


class LLMRequestError(LLMError):
    """The LLM endpoint rejected the request (4xx); retrying would not help."""


class LLMResponseError(LLMError):
    """The LLM endpoint returned 200 with a malformed OpenAI-style envelope."""


class ExecutionError(FinancialReportQAError):
    """Base class for Day 18 deterministic-compiler failures."""


class ExecutionInputError(ExecutionError):
    """A plan, release, or CLI input to the compiler is invalid."""


class ExecutionReplayMismatchError(ExecutionError):
    """A compiled `pandas_query` did not replay to the compiler's own answer."""


class ProgramError(ExecutionError):
    """Masked-PAL program handling failed (spec 2026-08-24)."""


class ProgramGuardError(ProgramError):
    """A generated program violated the N4' AST whitelist."""


class ProgramBindingError(ProgramError):
    """A candidate index could not be bound to a real cell."""


class ProgramEvalError(ProgramError):
    """A guarded program failed at evaluation time."""


class SubmissionError(FinancialReportQAError):
    """Base class for Day 22 submission bundle failures."""


class SubmissionInputError(SubmissionError):
    """A question file, ZIP, or CLI input to the submission bundle is invalid."""


class ExportError(FinancialReportQAError):
    """A CSV, metadata, or synced-text export artifact cannot be produced."""
