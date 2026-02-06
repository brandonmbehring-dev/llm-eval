"""llm-eval: Statistical RAG evaluation framework with drift detection.

Usage:
    from llm_eval import GoldenSet, GoldenQuery, EvalRun, RetrievalAdapter
    from llm_eval.runner import run_evaluation
    from llm_eval.metrics.ranking import mean_reciprocal_rank, ndcg_at_k
    from llm_eval.drift.detector import DriftDetector
"""

from llm_eval.adapter import RetrievalAdapter
from llm_eval.types import (
    Baseline,
    Difficulty,
    DriftResult,
    DriftSeverity,
    EvalRun,
    GoldenQuery,
    GoldenSet,
    QueryResult,
    RetrievedItem,
)

__all__ = [
    "Baseline",
    "Difficulty",
    "DriftResult",
    "DriftSeverity",
    "EvalRun",
    "GoldenQuery",
    "GoldenSet",
    "QueryResult",
    "RetrievalAdapter",
    "RetrievedItem",
]

__version__ = "0.1.0"
