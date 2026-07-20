from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.embeddings.base import EmbeddingProvider
from app.retrievers.base import Retriever
from app.retrievers.hybrid import HybridRetriever
from app.retrievers.postgres import LexicalPostgresRetriever, PgVectorRetriever
from app.retrievers.qdrant import QdrantRetriever
from app.schemas.search import DenseBackend, FusionMethod, RetrievalStrategy


def build_retriever(
    strategy: RetrievalStrategy,
    session: AsyncSession,
    qdrant_client: AsyncQdrantClient,
    settings: Settings,
    embedding_provider: EmbeddingProvider,
    dense_backend: DenseBackend,
    fusion_method: FusionMethod,
) -> Retriever:
    lexical = LexicalPostgresRetriever(session)
    pgvector = PgVectorRetriever(session, embedding_provider)
    qdrant = QdrantRetriever(qdrant_client, settings, embedding_provider)

    if strategy == RetrievalStrategy.lexical:
        return lexical
    if strategy == RetrievalStrategy.dense_pgvector:
        return pgvector
    if strategy == RetrievalStrategy.dense_qdrant:
        return qdrant
    if strategy == RetrievalStrategy.hybrid_pgvector:
        return HybridRetriever(lexical, pgvector, fusion_method.value)
    if strategy == RetrievalStrategy.hybrid_qdrant:
        return HybridRetriever(lexical, qdrant, fusion_method.value)

    dense: Retriever = pgvector if dense_backend == DenseBackend.pgvector else qdrant
    return HybridRetriever(lexical, dense, fusion_method.value)
