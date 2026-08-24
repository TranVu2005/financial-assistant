from __future__ import annotations

import re

from financial_report_qa.retrieval.dense_encoder import (
    approved_encoder_spec,
    encoder_spec_sha256,
)


def test_approved_encoder_specs_are_fully_pinned() -> None:
    """Changing an encoder revision or prefix must produce a different artifact identity."""
    bge = approved_encoder_spec("bge-m3")
    e5 = approved_encoder_spec("multilingual-e5-small")

    assert (bge.revision, bge.dimension, bge.query_prefix, bge.batch_size) == (
        "5617a9f61b028005a4858fdac845db406aefb181",
        1024,
        "",
        8,
    )
    assert (e5.revision, e5.dimension, e5.query_prefix, e5.document_prefix) == (
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        384,
        "query: ",
        "passage: ",
    )
    assert encoder_spec_sha256(e5) != encoder_spec_sha256(
        e5.model_copy(update={"query_prefix": ""})
    )
    assert bge.device == "cpu"
    assert encoder_spec_sha256(bge) != encoder_spec_sha256(
        bge.model_copy(update={"device": "cuda"})
    )


def test_qwen3_embedding_4b_is_an_approved_pinned_spec() -> None:
    spec = approved_encoder_spec("qwen3-embedding-4b")

    assert spec.model_id == "Qwen/Qwen3-Embedding-4B"
    assert re.fullmatch(r"[0-9a-f]{40}", spec.revision)
    assert spec.dimension == 2560
    assert spec.max_sequence_length == 8192
    # N5: điểm số liên tục dùng để xếp hạng -- không bao giờ quantize.
    assert spec.dtype == "float32"
    assert spec.normalize_embeddings is True


def test_legacy_encoders_keep_their_512_sequence_length() -> None:
    assert approved_encoder_spec("bge-m3").max_sequence_length == 512
    assert approved_encoder_spec("multilingual-e5-small").max_sequence_length == 512
