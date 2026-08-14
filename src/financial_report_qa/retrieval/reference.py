"""Versioned, content-bound retrieval reference identities."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from financial_report_qa.retrieval.evaluation import (
    RetrievalEvaluationReport,
    RetrievalMetrics,
    score_at_10,
)

ReferenceVersion = Literal["gold30", "gold70"]
_LOCKED_DATASET_FINGERPRINT = "422df141c935d46bfd14302abec50f32380e6e4c012159f8ad0ae5560c8a446a"

_GOLD30_QUESTION_IDS: tuple[str, ...] = (
    "retq_00888e79366b91100dacb03137b33c72c620c121dbf3e5fab1db36a23b41733e",
    "retq_027ca04462e4cc19229848df810cb6c6aa404ddd4b19659fee6cabd954fbcfd2",
    "retq_0a32a6d94a6e7bad8479d11ebbc10495710bc76f86ee2b0bde7d77462fa29d99",
    "retq_113ffd796a5be812ba4e774e0f114c81be109064417ec267cdeca9cc693885a0",
    "retq_1b05912ef66e0d457aaa4f6f1f6e9750bf1d63ca0680278ef53ccf5714858c40",
    "retq_276accff7b518a3d1b034d720a140950c4be2c4703534b5f7e130f3e3e2d29ab",
    "retq_46612a9c276fa94b4b682731803874a302616a3d4aaea063b36609e4603b7196",
    "retq_4ae188f21b74a920b9293a7898f06bc80be5d7c48c87e2b2cc6731194768ac42",
    "retq_4ed36641e810a4c72f8383ff9869cfa6cbccff95bef524f08fce369a24525cf4",
    "retq_5a293140dab6370835bb93b17bb0503467626e4386d5e5ca5264afd3d2cff41b",
    "retq_5adfcd4becee37f88a87d56f5a73cf8bd06c7a1ffea48f9f951c9997bdacafed",
    "retq_5e780dd26bf4c16168dc8f823b62918e50fe52d1acd0e673f1ac1a8bfc390dbd",
    "retq_6a6a6024dd19f425b10ad6fb6e58f1bf8842e1886363b812d379f54a35e44655",
    "retq_76c79261bdcb719e588c362ee794146de50e423a79e93c14949057684cc02dcf",
    "retq_799bb5213c176968fa13d78639d80e3e479c3ce0e6d1531912c7fc9dd0ffdf84",
    "retq_7afd1c8e800a18c317c7f2b540f77433d318a5b3aa2c483fc333294615416da2",
    "retq_8e43d1b9fd61ad81fb038d853c71c1c4b582ab7c502fae41ed33edf22737bc50",
    "retq_905e8568becf90520e588c3292c269b0a7030ba5ec4eff4f4cbcf8d21a0fba23",
    "retq_9490ea8fbfcc0834be9ca1d779411b8cf60ceeaf4816cbd05abc9c4ceeb48118",
    "retq_9abb0ade76092d5443f249b43fb913340f16cfc5318c8631869c939e065534bf",
    "retq_a2d5888138ba3e4af86938cf0854c85da519cf1d4c63f8e610fc722853f816a2",
    "retq_b241522bcdab166c95e5e24ad9fb63a9eede7baf7c8432392db31274c063c389",
    "retq_b48a824c13365c7b2537ece813ecc3f1e940d2765444b9b920a75418eb0e4a61",
    "retq_b74d8b5b9e878a98bf9431932d39659660d3ba35571478baa6e678ce932c4a45",
    "retq_bf660ff667f8d577c20962ceb66c154468642b12c827e7a52024b3fa2a5277bf",
    "retq_c04ab744e55a62eb20943d9d43f55c5fd18f88ecbd530a61cc4a0c818af4ff17",
    "retq_c069554491e65164bb302c37f0d9c83d0283546dbc61e26448e2cc5512cb06e2",
    "retq_d01701ab4c5e7d93a4af6f43f55ef17be41d4e2297131a4a388636edfdc8e8f2",
    "retq_daf3295706f9a99546fdaafdda237fc03c541ae74522a82b46bd39ed6d863bb4",
    "retq_ea3b207f9bcd90985977155a65aa00de9d9127eaa488d93394daa8e2c569ab71",
)


@dataclass(frozen=True)
class Bm25ReferenceDescriptor:
    """All immutable identities needed to distinguish one historical baseline."""

    version: ReferenceVersion
    dataset_fingerprint: str
    question_count: int
    gold_sha256: str
    question_ids_sha256: str
    report_sha256: str
    macro: RetrievalMetrics
    question_ids: tuple[str, ...] = ()


GOLD30_BM25_REFERENCE = Bm25ReferenceDescriptor(
    version="gold30",
    dataset_fingerprint=_LOCKED_DATASET_FINGERPRINT,
    question_count=30,
    gold_sha256="1b4646e5b2adac433522bfaa6d3de87951f0aef9a0600140336cd1ac65034404",
    question_ids_sha256="2a0e794c408f6f65d4acdc8fb5ff565f4254482fd6f76b22f5b272e9ea67e4c4",
    report_sha256="2653b2713fd34e032f2492cffcf530d1eb80285452bcd59c22097559a6ef1eed",
    macro=RetrievalMetrics(
        true_positive=44,
        precision=0.1466666666666667,
        recall=0.8833333333333333,
        f2=0.4312169312169313,
    ),
    question_ids=_GOLD30_QUESTION_IDS,
)

CURRENT_BM25_REFERENCE = Bm25ReferenceDescriptor(
    version="gold70",
    dataset_fingerprint=_LOCKED_DATASET_FINGERPRINT,
    question_count=70,
    gold_sha256="5ed12e6abfe03009a4792d45c2e437bbe615257fc2eeb20d8feb32ac9dbd8b9e",
    question_ids_sha256="d70de63b36af9af1017c14b87dcabd86c60ec186943eca772d540a0a5a66972a",
    report_sha256="75ebbc6ff68e4ad3d3ed104f2427f1e54489b7e7916941d06531a1c2da8877ad",
    macro=RetrievalMetrics(
        true_positive=109,
        precision=0.1557142857142856,
        recall=0.9142857142857143,
        f2=0.4377091162805452,
    ),
)

BM25_REFERENCES: tuple[Bm25ReferenceDescriptor, ...] = (
    GOLD30_BM25_REFERENCE,
    CURRENT_BM25_REFERENCE,
)


@dataclass(frozen=True)
class ResolvedBm25Reference:
    descriptor: Bm25ReferenceDescriptor
    report: RetrievalEvaluationReport


@dataclass(frozen=True)
class ResolvedGoldReference:
    descriptor: Bm25ReferenceDescriptor
    source_sha256: str
    selected_jsonl_sha256: str
    selected_question_ids: frozenset[str]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _question_ids_sha256(question_ids: tuple[str, ...]) -> str:
    return _sha256(("\n".join(question_ids) + "\n").encode())


def _metrics_match(observed: RetrievalMetrics, expected: RetrievalMetrics) -> bool:
    return observed.true_positive == expected.true_positive and all(
        math.isclose(left, right, abs_tol=5e-8, rel_tol=0.0)
        for left, right in (
            (observed.precision, expected.precision),
            (observed.recall, expected.recall),
            (observed.f2, expected.f2),
        )
    )


def _average_metrics(values: tuple[RetrievalMetrics, ...]) -> RetrievalMetrics:
    count = len(values)
    return RetrievalMetrics(
        true_positive=sum(item.true_positive for item in values),
        precision=sum(item.precision for item in values) / count,
        recall=sum(item.recall for item in values) / count,
        f2=sum(item.f2 for item in values) / count,
    )


def validate_bm25_reference_report(
    report: RetrievalEvaluationReport,
) -> Bm25ReferenceDescriptor:
    """Identify and deeply validate a complete gold30 or gold70 BM25 report."""
    if report.question_count != len(report.per_question):
        raise ValueError("BM25 reference question_count must equal per_question length")
    question_ids = tuple(item.question_id for item in report.per_question)
    if question_ids != tuple(sorted(set(question_ids))):
        raise ValueError("BM25 reference question IDs must be sorted and unique")
    question_ids_sha256 = _question_ids_sha256(question_ids)
    descriptor = next(
        (
            item
            for item in BM25_REFERENCES
            if report.dataset_fingerprint == item.dataset_fingerprint
            and report.question_count == item.question_count
            and question_ids_sha256 == item.question_ids_sha256
            and _metrics_match(report.macro, item.macro)
        ),
        None,
    )
    if descriptor is None:
        raise ValueError("BM25 reference does not match a locked gold30/gold70 identity")

    rescored = tuple(
        score_at_10(item.predicted_table_ids, item.gold_table_ids) for item in report.per_question
    )
    if any(
        not _metrics_match(observed, expected)
        for observed, expected in zip(
            (item.metrics for item in report.per_question), rescored, strict=True
        )
    ):
        raise ValueError("BM25 reference per-question metrics do not match its rankings")
    recomputed_macro = _average_metrics(rescored)
    if not _metrics_match(report.macro, recomputed_macro):
        raise ValueError("BM25 reference macro does not match per-question metrics")
    by_intent_values: dict[str, list[RetrievalMetrics]] = defaultdict(list)
    for item, metrics in zip(report.per_question, rescored, strict=True):
        by_intent_values[item.intent].append(metrics)
    recomputed_by_intent = {
        intent: _average_metrics(tuple(by_intent_values[intent]))
        for intent in sorted(by_intent_values)
    }
    if set(report.by_intent) != set(recomputed_by_intent) or any(
        not _metrics_match(report.by_intent[intent], metrics)
        for intent, metrics in recomputed_by_intent.items()
    ):
        raise ValueError("BM25 reference by_intent does not match per-question metrics")
    return descriptor


def load_bm25_reference_report(path: Path) -> ResolvedBm25Reference:
    """Load only an exact deterministic reference artifact."""
    raw = path.read_bytes()
    artifact_sha256 = _sha256(raw)
    descriptor = next(
        (item for item in BM25_REFERENCES if item.report_sha256 == artifact_sha256), None
    )
    if descriptor is None:
        raise ValueError("BM25 reference artifact SHA-256 is not locked")
    report = RetrievalEvaluationReport.model_validate_json(raw)
    observed = validate_bm25_reference_report(report)
    if observed != descriptor:
        raise ValueError("BM25 reference artifact content does not match its descriptor")
    return ResolvedBm25Reference(descriptor=descriptor, report=report)


def resolve_gold_reference(path: Path, *, version: ReferenceVersion) -> ResolvedGoldReference:
    """Resolve gold70 or its byte-identical historical gold30 subset by content."""
    descriptor = next(item for item in BM25_REFERENCES if item.version == version)
    raw = path.read_bytes()
    source_sha256 = _sha256(raw)
    if version == "gold70":
        if source_sha256 != descriptor.gold_sha256:
            raise ValueError("gold70 artifact SHA-256 does not match the current reference")
        selected_raw = raw
    elif source_sha256 == descriptor.gold_sha256:
        selected_raw = raw
    elif source_sha256 == CURRENT_BM25_REFERENCE.gold_sha256:
        selected_ids = set(descriptor.question_ids)
        try:
            selected_raw = b"".join(
                line
                for line in raw.splitlines(keepends=True)
                if json.loads(line)["question_id"] in selected_ids
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("gold reference JSONL cannot be resolved") from exc
    else:
        raise ValueError("gold artifact SHA-256 does not match gold30 or current gold70")

    selected_jsonl_sha256 = _sha256(selected_raw)
    if selected_jsonl_sha256 != descriptor.gold_sha256:
        raise ValueError("selected historical gold JSONL does not match its locked SHA-256")
    try:
        selected_question_ids = tuple(
            json.loads(line)["question_id"] for line in selected_raw.splitlines()
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("selected gold reference JSONL is invalid") from exc
    if (
        len(selected_question_ids) != descriptor.question_count
        or _question_ids_sha256(selected_question_ids) != descriptor.question_ids_sha256
    ):
        raise ValueError("selected gold question IDs do not match the locked reference")
    return ResolvedGoldReference(
        descriptor=descriptor,
        source_sha256=source_sha256,
        selected_jsonl_sha256=selected_jsonl_sha256,
        selected_question_ids=frozenset(selected_question_ids),
    )
