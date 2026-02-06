"""Optional research-kb adapter for llm-eval.

Provides ResearchKBAdapter that wraps the research-kb hybrid search.

Usage:
    from llm_eval_research_kb import ResearchKBAdapter

    adapter = ResearchKBAdapter()
    run = run_evaluation(golden_set, adapter)
"""

from llm_eval_research_kb.adapter import ResearchKBAdapter

__all__ = ["ResearchKBAdapter"]
