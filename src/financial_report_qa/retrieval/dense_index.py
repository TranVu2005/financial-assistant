"""Exact FAISS dense-index construction and integrity checks."""

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

from financial_report_qa.core.errors import DenseArtifactError
from financial_report_qa.retrieval.dense_artifacts import file_sha256, write_text_atomic
from financial_report_qa.retrieval.dense_contracts import DenseIndexManifest
from financial_report_qa.retrieval.dense_corpus import DenseCorpus
from financial_report_qa.retrieval.dense_encoder import DenseEncoder, encoder_spec_sha256

ProgressCallback = Callable[[int, int, float], None]


@dataclass(frozen=True)
class DenseIndex:
    corpus: DenseCorpus
    faiss_index: faiss.IndexFlatIP
    manifest: DenseIndexManifest


def _vectors(values: np.ndarray, rows: int, dimension: int) -> np.ndarray:
    if values.dtype != np.float32 or values.shape != (rows, dimension):
        raise ValueError("dense encoder returned an invalid float32 shape")
    if not np.isfinite(values).all() or not np.allclose(
        np.linalg.norm(values, axis=1), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError("dense encoder returned invalid normalized vectors")
    return np.ascontiguousarray(values)


def build_dense_index(
    corpus: DenseCorpus,
    encoder: DenseEncoder,
    *,
    faiss_device: Literal["cpu", "cuda"] = "cpu",
    progress: ProgressCallback | None = None,
) -> DenseIndex:
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
    return DenseIndex(
        corpus,
        cast(faiss.IndexFlatIP, index),
        DenseIndexManifest(
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


def save_dense_index(index: DenseIndex, output_dir: Path) -> Path:
    if output_dir.exists():
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


def load_dense_index(
    output_dir: Path,
    corpus: DenseCorpus,
    *,
    expected_encoder_spec_sha256: str,
    release_lock_sha256: str,
) -> DenseIndex:
    payload = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest = DenseIndexManifest.model_validate(payload)
    path = output_dir / "index.faiss"
    if (
        manifest.release_lock_sha256 != release_lock_sha256
        or manifest.document_sha256 != corpus.manifest.document_sha256
        or manifest.encoder_spec_sha256 != expected_encoder_spec_sha256
        or manifest.artifact_sha256.get("index.faiss") != file_sha256(path)
    ):
        raise DenseArtifactError("Dense index identity or artifact hash does not match")
    loaded = faiss.read_index(str(path))
    if (
        not isinstance(loaded, faiss.IndexFlatIP)
        or loaded.d != manifest.dimension
        or loaded.ntotal != manifest.document_count
    ):
        raise DenseArtifactError("Dense FAISS index does not match manifest")
    return DenseIndex(corpus, loaded, manifest)
