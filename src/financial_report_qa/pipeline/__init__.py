"""Day 21 end-to-end pipeline: retrieval -> plan -> execution -> verification.

See `pipeline.evaluation.run_e2e_pipeline` for the entry point and
[ADR 0010](../../../docs/decisions/0010-statement-scope-contract.md) for why
this module exists separately from `verification/evaluation.py` (which
intentionally measures compiler+verifier under perfect retrieval, not the
real system).
"""
