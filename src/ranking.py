import math


def make_final_ranking(
    declaration_text: str,
    candidate_indices: list[int],
    regulation_texts: list[str],
    reranker,
    top_k: int = 10,
):
    candidates = [
        (idx, regulation_texts[idx])
        for idx in candidate_indices
    ]

    ranked = reranker.rerank(
        declaration_text,
        candidates,
        top_k=top_k,
    )

    return ranked


def validate_top_k(
    ranked: list[tuple[int, float]],
    top_k: int = 10,
):
    if len(ranked) != top_k:
        raise ValueError(
            f"Expected {top_k} results, got {len(ranked)}"
        )

    indices = [idx for idx, _ in ranked]

    if len(set(indices)) != len(indices):
        raise ValueError("Duplicate regulation indices")

    for _, score in ranked:
        if not math.isfinite(score):
            raise ValueError(f"Non-finite score: {score}")