from typing import Protocol

from app.schemas.search import SearchResult


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]: ...
