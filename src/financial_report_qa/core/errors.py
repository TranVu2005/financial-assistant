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
