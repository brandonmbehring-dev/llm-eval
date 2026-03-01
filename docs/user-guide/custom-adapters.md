# Custom Adapters

Adapters connect ir-eval to your retrieval system for live evaluation. They implement the `RetrievalAdapter` protocol.

## The Protocol

```python
from ir_eval import RetrievalAdapter, RetrievedItem

class MyAdapter(RetrievalAdapter):
    @property
    def name(self) -> str:
        return "my-system"

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        # Call your retrieval system
        results = my_search(query, limit=top_k)
        return [
            RetrievedItem(
                id=r["id"],
                rank=i + 1,
                score=r["score"],
                content=r.get("text"),
                metadata=r.get("metadata"),
            )
            for i, r in enumerate(results)
        ]
```

## Design Decision: Sync Protocol

`retrieve()` is synchronous by design. Most retrieval systems (Elasticsearch, LangChain, LlamaIndex) have sync APIs. Async systems wrap internally:

```python
import asyncio

class AsyncAdapter(RetrievalAdapter):
    @property
    def name(self) -> str:
        return "async-system"

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        return asyncio.run(self._async_retrieve(query, top_k))

    async def _async_retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        results = await my_async_search(query, limit=top_k)
        return [RetrievedItem(id=r.id, rank=i+1, score=r.score) for i, r in enumerate(results)]
```

## Registration via Entry Points

Register your adapter as a Python entry point for CLI discovery:

```toml
# pyproject.toml
[project.entry-points."ir_eval.adapters"]
my-system = "my_package:MyAdapter"
```

Then use from CLI:

```bash
ir-eval run golden.json --adapter my-system
```

## Example: Research-KB Adapter

The `ir_eval_research_kb` package provides a reference adapter implementation:

```python
class ResearchKBAdapter(RetrievalAdapter):
    """Adapter for research-kb MCP server."""

    @property
    def name(self) -> str:
        return "research-kb"

    def retrieve(self, query: str, top_k: int) -> list[RetrievedItem]:
        # Queries research-kb via MCP protocol
        chunks = self.client.search(query, limit=top_k)
        return [
            RetrievedItem(
                id=chunk.id,
                rank=i + 1,
                score=chunk.score,
                content=chunk.text,
                metadata={"source": chunk.source},
            )
            for i, chunk in enumerate(chunks)
        ]
```

## Testing Adapters

```python
def test_adapter_protocol():
    adapter = MyAdapter()
    assert isinstance(adapter, RetrievalAdapter)
    assert adapter.name == "my-system"

    results = adapter.retrieve("test query", top_k=5)
    assert len(results) <= 5
    assert all(isinstance(r, RetrievedItem) for r in results)
```
