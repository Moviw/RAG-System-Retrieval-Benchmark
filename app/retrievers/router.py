import re
from dataclasses import dataclass

from app.retrievers.base import Retriever
from app.schemas.search import SearchResult

ERROR_CODE_RE = re.compile(r"\b[A-Z]{2,10}-\d{2,6}\b")
VERSION_RE = re.compile(r"\b(v?\d+\.\d+(?:\.\d+)?)\b")
FUNCTION_RE = re.compile(r"\b[a-zA-Z_][\w]*\.[a-zA-Z_][\w]*\b|\b[a-zA-Z_][\w]*\(\)")
PATH_RE = re.compile(r"(/[\w./-]+)|(\b[\w.-]+/[\w./-]+)")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)


@dataclass(frozen=True)
class QueryFeatures:
    has_exact_identifier: bool
    has_error_code: bool
    has_version: bool
    has_function_name: bool
    has_path: bool
    has_uuid: bool
    token_count: int
    question_like: bool


@dataclass(frozen=True)
class RoutingDecision:
    strategy: str
    reason: str
    features: QueryFeatures


def extract_query_features(query: str) -> QueryFeatures:
    tokens = query.split()
    has_error_code = bool(ERROR_CODE_RE.search(query))
    has_version = bool(VERSION_RE.search(query))
    has_function_name = bool(FUNCTION_RE.search(query))
    has_path = bool(PATH_RE.search(query))
    has_uuid = bool(UUID_RE.search(query))
    has_exact_identifier = (
        has_error_code or has_version or has_function_name or has_path or has_uuid
    )
    question_like = (
        query.lower().startswith(("what", "how", "why", "find", "explain")) or "?" in query
    )
    return QueryFeatures(
        has_exact_identifier=has_exact_identifier,
        has_error_code=has_error_code,
        has_version=has_version,
        has_function_name=has_function_name,
        has_path=has_path,
        has_uuid=has_uuid,
        token_count=len(tokens),
        question_like=question_like,
    )


class RuleBasedQueryRouter:
    def route(self, query: str) -> RoutingDecision:
        features = extract_query_features(query)
        if features.has_exact_identifier and not features.question_like:
            return RoutingDecision("lexical", "identifier-heavy query", features)
        if features.has_exact_identifier and features.question_like:
            return RoutingDecision("hybrid", "identifier plus natural-language intent", features)
        if features.question_like or features.token_count >= 6:
            return RoutingDecision("dense", "semantic natural-language query", features)
        return RoutingDecision("hybrid", "short ambiguous query", features)


class AdaptiveRouterRetriever:
    def __init__(
        self,
        lexical: Retriever,
        dense: Retriever,
        hybrid: Retriever,
        router: RuleBasedQueryRouter | None = None,
    ) -> None:
        self.lexical = lexical
        self.dense = dense
        self.hybrid = hybrid
        self.router = router or RuleBasedQueryRouter()
        self.last_decision: RoutingDecision | None = None

    async def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[SearchResult]:
        decision = self.router.route(query)
        self.last_decision = decision
        if decision.strategy == "lexical":
            results = await self.lexical.search(query, top_k, filters)
        elif decision.strategy == "dense":
            results = await self.dense.search(query, top_k, filters)
        else:
            results = await self.hybrid.search(query, top_k, filters)
        return [
            result.model_copy(
                update={
                    "strategy": f"adaptive_{decision.strategy}",
                    "metadata": {
                        **result.metadata,
                        "routing_decision": decision.strategy,
                        "routing_reason": decision.reason,
                    },
                }
            )
            for result in results
        ]
