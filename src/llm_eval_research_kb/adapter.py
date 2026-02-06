"""Research-kb retrieval adapter for llm-eval.

Wraps the research-kb hybrid search (FTS + vector) as a synchronous
RetrievalAdapter. Uses asyncio.run() internally since research-kb
is async.

Requirements:
    - research-kb packages on PYTHONPATH (storage, pdf-tools, common, contracts)
    - PostgreSQL with research-kb data
    - BGE-large embedding model accessible

Usage:
    from llm_eval_research_kb import ResearchKBAdapter
    from llm_eval.runner import run_evaluation

    adapter = ResearchKBAdapter()
    run = run_evaluation(golden_set, adapter, top_k=10)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from llm_eval.types import RetrievedItem


class ResearchKBAdapter:
    """Adapter bridging research-kb search to llm-eval.

    Args:
        fts_weight: Full-text search weight (default 0.3).
        vector_weight: Vector similarity weight (default 0.7).
        research_kb_root: Path to research-kb repo root.
    """

    def __init__(
        self,
        fts_weight: float = 0.3,
        vector_weight: float = 0.7,
        research_kb_root: Path | None = None,
    ) -> None:
        self._fts_weight = fts_weight
        self._vector_weight = vector_weight
        self._initialized = False

        # Add research-kb packages to path
        root = research_kb_root or Path.home() / "Claude" / "research-kb"
        for pkg in ["pdf-tools", "storage", "common", "contracts"]:
            pkg_path = str(root / "packages" / pkg / "src")
            if pkg_path not in sys.path:
                sys.path.insert(0, pkg_path)

    def _ensure_init(self) -> None:
        """Lazy-initialize async resources."""
        if not self._initialized:
            from research_kb_pdf import EmbeddingClient  # type: ignore[import-not-found]
            from research_kb_storage import (  # type: ignore[import-not-found]
                DatabaseConfig,
                get_connection_pool,
            )

            self._embed_client = EmbeddingClient()
            self._db_config = DatabaseConfig()

            async def _init() -> None:
                await get_connection_pool(self._db_config)

            asyncio.run(_init())
            self._initialized = True

    @property
    def name(self) -> str:
        """Adapter name for reports."""
        return "research-kb"

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievedItem]:
        """Execute hybrid search against research-kb.

        Args:
            query: Search query text.
            top_k: Number of results.

        Returns:
            Ranked list of RetrievedItem.
        """
        self._ensure_init()

        from research_kb_storage import SearchQuery, search_hybrid  # noqa: F811

        query_embedding = self._embed_client.embed_query(query)

        search_query = SearchQuery(
            text=query,
            embedding=query_embedding,
            fts_weight=self._fts_weight,
            vector_weight=self._vector_weight,
            limit=top_k,
        )

        results = asyncio.run(search_hybrid(search_query))

        return [
            RetrievedItem(
                id=str(r.chunk.id),
                rank=r.rank,
                score=getattr(r, "score", None),
                content=getattr(r.chunk, "content", None),
                metadata={
                    "source_title": r.source.title,
                    "page_start": getattr(r.chunk, "page_start", None),
                },
            )
            for r in results
        ]
