from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    chunk_index: int
    content: str
    token_count: int


def tokenize(text: str) -> list[str]:
    return text.split()


def chunk_text(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[TextChunk]:
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be non-negative")
    if overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")

    tokens = tokenize(text)
    if not tokens:
        return []

    chunks: list[TextChunk] = []
    start = 0
    index = 0
    step = chunk_size_tokens - overlap_tokens
    while start < len(tokens):
        window = tokens[start : start + chunk_size_tokens]
        chunks.append(
            TextChunk(chunk_index=index, content=" ".join(window), token_count=len(window))
        )
        if start + chunk_size_tokens >= len(tokens):
            break
        start += step
        index += 1
    return chunks
