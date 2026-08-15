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

# Day 21 plan §1.9: retrieval-gold-v1.jsonl grew 70 -> 120 (50 questions
# appended, the original 70 kept byte-identical). `CURRENT_BM25_REFERENCE`
# below is no longer literally the whole live file -- these ids let it be
# reconstructed as a subset, exactly like `_GOLD30_QUESTION_IDS` already is.
_GOLD70_QUESTION_IDS: tuple[str, ...] = (
    "retq_00888e79366b91100dacb03137b33c72c620c121dbf3e5fab1db36a23b41733e",
    "retq_027ca04462e4cc19229848df810cb6c6aa404ddd4b19659fee6cabd954fbcfd2",
    "retq_035ab7b50aa16f2da0835e329105ef45254fff823aab0375464fe935992d7301",
    "retq_07495f696860ec468fa5a71f67fa84b86bcfdbcd1b55d1deaf3fbbbebf45a52b",
    "retq_0a32a6d94a6e7bad8479d11ebbc10495710bc76f86ee2b0bde7d77462fa29d99",
    "retq_0eaaf1a9804038a40987869bdcc2da226dbe3aba08d02943873b029d4f172848",
    "retq_113ffd796a5be812ba4e774e0f114c81be109064417ec267cdeca9cc693885a0",
    "retq_140b9a346d497feed47dd86d6eb1a2bb0fa9c2c4ef873306cad2b631a8e347ce",
    "retq_15f357970d80fc9aa487f0da8d252b505d1217d23c365019000c60a366d1f620",
    "retq_184e75fa031778165edf54a7777b54c7cf874c7b1df60857be094b591c2564f1",
    "retq_18d416f9aaed3300ef83d93fd5743112c68fc2f6a11d518cc4f029be24e3766a",
    "retq_1b05912ef66e0d457aaa4f6f1f6e9750bf1d63ca0680278ef53ccf5714858c40",
    "retq_21d0fcc00dee1bd24d014fda92df6768482761e46ff4d1b809e4a8a61a8f2e58",
    "retq_276accff7b518a3d1b034d720a140950c4be2c4703534b5f7e130f3e3e2d29ab",
    "retq_2bf1a7ff817063f8a7583a5809b021257faa2bf25252135d085d38ffeb21a84f",
    "retq_2c6e03ad7f1a391d3673e2d8148a083ee759a1c1149c6f379241323f8b20221d",
    "retq_3311745c98ba5efc0f6b7e78f74fbb69d061ec16f6f266fa0d82f83b1564dee0",
    "retq_3e6c4ae3ff4d30628320182dce41772a231e92f8cb6d740a476bccdee64f5b06",
    "retq_46612a9c276fa94b4b682731803874a302616a3d4aaea063b36609e4603b7196",
    "retq_4ae188f21b74a920b9293a7898f06bc80be5d7c48c87e2b2cc6731194768ac42",
    "retq_4af1d194037feec883a2ad351fad2d857c5a714df82de48c323b5b03da976fd4",
    "retq_4ed36641e810a4c72f8383ff9869cfa6cbccff95bef524f08fce369a24525cf4",
    "retq_518fddfe44aa2ba59c07c69fba10ccc85837f034cd9bd0f31802eefd97757c7c",
    "retq_5a293140dab6370835bb93b17bb0503467626e4386d5e5ca5264afd3d2cff41b",
    "retq_5adfcd4becee37f88a87d56f5a73cf8bd06c7a1ffea48f9f951c9997bdacafed",
    "retq_5bc82c57026f9a9c465c6de9a5423aa4dcc47c3bcb909980e2a8b34312c723e5",
    "retq_5e5a7482abba820a4e8819833d85d90cbb45fcd33bd23e5214a29f9f4e861f5d",
    "retq_5e780dd26bf4c16168dc8f823b62918e50fe52d1acd0e673f1ac1a8bfc390dbd",
    "retq_66ed383b79241da41bd2eee0070530fc2e90964b213e9bcaa3d7429b57c2826b",
    "retq_6a6a6024dd19f425b10ad6fb6e58f1bf8842e1886363b812d379f54a35e44655",
    "retq_70fa3ba240c02f67e25d7f74451a303b71bdab96a0a660807d3f1d48621e2318",
    "retq_76c79261bdcb719e588c362ee794146de50e423a79e93c14949057684cc02dcf",
    "retq_799bb5213c176968fa13d78639d80e3e479c3ce0e6d1531912c7fc9dd0ffdf84",
    "retq_7a23e39cc00a015db310ae9865241133d6c2bebebb760ab4a9b0341c96de6628",
    "retq_7afd1c8e800a18c317c7f2b540f77433d318a5b3aa2c483fc333294615416da2",
    "retq_7d8b6d321910394b5f0eaeeb22a747545e1d297ae6f08f8a3f9243c11c557125",
    "retq_843c4f4d5dbc6284787d601a04e41c8bd5e98f2374d96a5432d69ad85daedbe8",
    "retq_859f3db5be0b80c43b04417d2369df915845fd9cfec70503dbe753f23ea43fe6",
    "retq_865b76cc9a17d3afd764551b50a9de5b68384b62c2366a2273804e80d1d999a1",
    "retq_8e43d1b9fd61ad81fb038d853c71c1c4b582ab7c502fae41ed33edf22737bc50",
    "retq_9024e98663eb4d6d935ed2bd5c014bdeff9ba0d94a7e62038833deb6ff1b464a",
    "retq_905e8568becf90520e588c3292c269b0a7030ba5ec4eff4f4cbcf8d21a0fba23",
    "retq_940e4706d5eea84a3bc3a90d109ddc7298d497fb90cbf4899fb043edb2268064",
    "retq_9490ea8fbfcc0834be9ca1d779411b8cf60ceeaf4816cbd05abc9c4ceeb48118",
    "retq_9a05b36a4b132789fe7ddd37f32f3ea8f6d14b2f51bef2ca73991c166adfa1d5",
    "retq_9abb0ade76092d5443f249b43fb913340f16cfc5318c8631869c939e065534bf",
    "retq_9cf253d87eebc6cff165ab32f1d759d9779d11bd67331bb56806fcf6af78a8c4",
    "retq_9ecbb4727d458b5de69ed6ff3355d237b4482e86765f3b9d339c37f86c66c3d7",
    "retq_a2d5888138ba3e4af86938cf0854c85da519cf1d4c63f8e610fc722853f816a2",
    "retq_a51c4303ed6e770889e289f07ef1704ee7ccd896f35768a69a203118062b9e16",
    "retq_a5af50731f92719510085938eae0f29f7cb39b3b0457e15e99748f5f61a22a86",
    "retq_a92e064c27aea88ce6b2d4d81bae32e94fc23ac351036e8fd88058ed484d3d62",
    "retq_af86db7ee5e169a65e2211bd3a20af63c4ecd906bf97d4572b15d0b6ea01859b",
    "retq_b241522bcdab166c95e5e24ad9fb63a9eede7baf7c8432392db31274c063c389",
    "retq_b48a824c13365c7b2537ece813ecc3f1e940d2765444b9b920a75418eb0e4a61",
    "retq_b74d8b5b9e878a98bf9431932d39659660d3ba35571478baa6e678ce932c4a45",
    "retq_bf660ff667f8d577c20962ceb66c154468642b12c827e7a52024b3fa2a5277bf",
    "retq_c04ab744e55a62eb20943d9d43f55c5fd18f88ecbd530a61cc4a0c818af4ff17",
    "retq_c069554491e65164bb302c37f0d9c83d0283546dbc61e26448e2cc5512cb06e2",
    "retq_c3b64537544ba45a111dd7a0aced43190d1073ed76f5927986e770e74c0c08b5",
    "retq_d01701ab4c5e7d93a4af6f43f55ef17be41d4e2297131a4a388636edfdc8e8f2",
    "retq_d18b935bc13fd5a2ddaae8ac584243d78c0d55b66468f26d18e8d92bd5af1dcd",
    "retq_d232835d75041d3a6a2f1b8c143f5cb34e83df0ce9dbfd82e70510b073045214",
    "retq_daf3295706f9a99546fdaafdda237fc03c541ae74522a82b46bd39ed6d863bb4",
    "retq_de3c339b336e378295c8a5298d140ca41aa623690c6aed3616d4b9abc3dbe919",
    "retq_df6f68634d600a5378892f46c619a80464948abd11da8a233fe225e749590992",
    "retq_e2059ca4277998aa17e8e41c3881e93960b982416b06a73b520628e942e263d5",
    "retq_ea3b207f9bcd90985977155a65aa00de9d9127eaa488d93394daa8e2c569ab71",
    "retq_ea502802099c055da44f3cf5e1352444a2df08eed91049e1a8e463f3277ba886",
    "retq_f4d7995ffd78410b783739fab33df211f00ce268cea4a5403c2b33e0ccb7832f",
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
    question_ids=_GOLD70_QUESTION_IDS,
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
    """Resolve gold30/gold70, either byte-identical or as a subset of a
    larger live file (Day 21 plan §1.9: retrieval-gold-v1.jsonl grew 70 ->
    120, keeping the original 70 byte-identical but no longer equal to the
    whole file) -- both are just "the locked question_ids, extracted from
    whatever is on disk now.
    """
    descriptor = next(item for item in BM25_REFERENCES if item.version == version)
    raw = path.read_bytes()
    source_sha256 = _sha256(raw)
    if source_sha256 == descriptor.gold_sha256:
        selected_raw = raw
    else:
        selected_ids = set(descriptor.question_ids)
        try:
            selected_raw = b"".join(
                line
                for line in raw.splitlines(keepends=True)
                if json.loads(line)["question_id"] in selected_ids
            )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("gold reference JSONL cannot be resolved") from exc

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
