"""Exact FAISS row-dense-index construction and integrity checks."""

from __future__ import annotations

import json
import platform
import shutil
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Literal, cast

import faiss
import numpy as np
import sentence_transformers
from pydantic import Field

from financial_report_qa.core.errors import DenseArtifactError
from financial_report_qa.retrieval.contracts import Fingerprint, NonEmptyString, _FrozenModel
from financial_report_qa.retrieval.dense_artifacts import file_sha256, write_text_atomic
from financial_report_qa.retrieval.dense_contracts import DenseEncoderSpec
from financial_report_qa.retrieval.dense_encoder import DenseEncoder, encoder_spec_sha256
from financial_report_qa.retrieval.row_dense_corpus import RowDenseCorpus


class RowDenseIndexManifest(_FrozenModel):
    """Identity and integrity data for one exact FAISS row dense index."""

    schema_version: Literal["row-dense-index-v1"] = "row-dense-index-v1"
    builder_version: Literal["v1"] = "v1"
    dataset_fingerprint: Fingerprint
    release_lock_sha256: Fingerprint
    document_sha256: Fingerprint
    encoder: DenseEncoderSpec
    encoder_spec_sha256: Fingerprint
    document_count: int = Field(ge=0)
    dimension: int = Field(gt=0)
    index_type: Literal["IndexFlatIP"] = "IndexFlatIP"
    metric: Literal["inner_product"] = "inner_product"
    dtype: Literal["float32"] = "float32"
    normalized: Literal[True] = True
    index_byte_size: int = Field(ge=0)
    library_versions: dict[str, NonEmptyString]
    artifact_sha256: dict[str, Fingerprint] = Field(default_factory=dict)


ProgressCallback = Callable[[int, int, float], None]


@dataclass(frozen=True)
class RowDenseIndex:
    corpus: RowDenseCorpus
    faiss_index: faiss.IndexFlatIP
    manifest: RowDenseIndexManifest


def _vectors(values: np.ndarray, rows: int, dimension: int) -> np.ndarray:
    if values.dtype != np.float32 or values.shape != (rows, dimension):
        raise ValueError("dense encoder returned an invalid float32 shape")
    if not np.isfinite(values).all() or not np.allclose(
        np.linalg.norm(values, axis=1), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("dense encoder returned invalid normalized vectors")
    return np.ascontiguousarray(values)


def build_row_dense_index(
    corpus: RowDenseCorpus,
    encoder: DenseEncoder,
    *,
    faiss_device: Literal["cpu", "cuda"] = "cpu",
    progress: ProgressCallback | None = None,
) -> RowDenseIndex:
    cpu_index = faiss.IndexFlatIP(encoder.spec.dimension)
    index: faiss.Index = cpu_index
    if faiss_device == "cpu":
        pass
    elif faiss_device == "cuda":
        required_apis = ("StandardGpuResources", "index_cpu_to_gpu", "index_gpu_to_cpu")
        missing_apis = [name for name in required_apis if not callable(getattr(faiss, name, None))]
        get_num_gpus = getattr(faiss, "get_num_gpus", None)
        if not callable(get_num_gpus):
            missing_apis.append("get_num_gpus")
        if missing_apis:
            raise DenseArtifactError(
                f"FAISS CUDA support unavailable: missing APIs: {', '.join(missing_apis)}"
            )
        assert callable(get_num_gpus)
        if get_num_gpus() <= 0:
            raise DenseArtifactError(
                "FAISS CUDA support unavailable: faiss.get_num_gpus() returned 0"
            )
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
    else:
        raise DenseArtifactError(f"Unsupported FAISS device: {faiss_device}")

    encoded = 0
    started_at = monotonic()
    for start in range(0, len(corpus.documents), encoder.spec.batch_size):
        documents = corpus.documents[start : start + encoder.spec.batch_size]
        index.add(
            _vectors(
                encoder.encode_documents(tuple(item.text for item in documents)),
                len(documents),
                encoder.spec.dimension,
            )
        )
        encoded += len(documents)
        if progress is not None:
            progress(encoded, len(corpus.documents), monotonic() - started_at)
    if faiss_device == "cuda":
        index = cast(faiss.IndexFlatIP, faiss.index_gpu_to_cpu(index))
    return RowDenseIndex(
        corpus,
        cast(faiss.IndexFlatIP, index),
        RowDenseIndexManifest(
            dataset_fingerprint=corpus.manifest.dataset_fingerprint,
            release_lock_sha256=corpus.manifest.release_lock_sha256,
            document_sha256=corpus.manifest.document_sha256,
            encoder=encoder.spec,
            encoder_spec_sha256=encoder_spec_sha256(encoder.spec),
            document_count=len(corpus.documents),
            dimension=encoder.spec.dimension,
            index_byte_size=0,
            library_versions={
                "faiss": faiss.__version__,
                "numpy": np.__version__,
                "sentence_transformers": sentence_transformers.__version__,
                "python": platform.python_version(),
            },
        ),
    )


def _dense_index_identity(manifest: RowDenseIndexManifest) -> dict[str, object]:
    return manifest.model_dump(mode="json", exclude={"artifact_sha256", "index_byte_size"})


def save_row_dense_index(index: RowDenseIndex, output_dir: Path) -> Path:
    """Publish atomically; reject an existing non-identical content-addressed target."""
    if output_dir.exists():
        existing_payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        existing_manifest = RowDenseIndexManifest.model_validate(existing_payload)
        if _dense_index_identity(existing_manifest) != _dense_index_identity(index.manifest):
            raise DenseArtifactError(
                f"Row dense index target already exists with different content: {output_dir}"
            )
        return output_dir
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        path = temporary / "index.faiss"
        faiss.write_index(index.faiss_index, str(path))
        manifest = index.manifest.model_copy(
            update={
                "index_byte_size": path.stat().st_size,
                "artifact_sha256": {"index.faiss": file_sha256(path)},
            }
        )
        write_text_atomic(
            temporary / "manifest.json",
            json.dumps(
                manifest.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True
            )
            + "\n",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_dir


def load_row_dense_index(
    output_dir: Path,
    corpus: RowDenseCorpus,
    *,
    expected_encoder_spec_sha256: str,
    release_lock_sha256: str,
) -> RowDenseIndex:
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = RowDenseIndexManifest.model_validate(payload)
    path = output_dir / "index.faiss"
    if (
        manifest.release_lock_sha256 != release_lock_sha256
        or manifest.document_sha256 != corpus.manifest.document_sha256
        or manifest.encoder_spec_sha256 != expected_encoder_spec_sha256
        or manifest.artifact_sha256.get("index.faiss") != file_sha256(path)
    ):
        raise DenseArtifactError("Row dense index identity or artifact hash does not match")
    loaded = faiss.read_index(str(path))
    if (
        not isinstance(loaded, faiss.IndexFlatIP)
        or loaded.d != manifest.dimension
        or loaded.ntotal != manifest.document_count
    ):
        raise DenseArtifactError("Row dense FAISS index does not match manifest")
    return RowDenseIndex(corpus, loaded, manifest)
