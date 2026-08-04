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
