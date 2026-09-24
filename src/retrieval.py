from collections import defaultdict


def reciprocal_rank_fusion(
    rankings: list[list[tuple[int, float]]],
    k: int = 60,
    top_k: int = 100,
) -> list[tuple[int, float]]:
    """
    Reciprocal Rank Fusion.

    rankings:
        несколько списков вида:
        [(doc_idx, score), ...]

    Исходные score здесь не используются.
    Важен только rank.
    """

    fused_scores = defaultdict(float)

    for ranking in rankings:
        for rank, (doc_idx, _) in enumerate(ranking, start=1):
            fused_scores[doc_idx] += 1.0 / (k + rank)

    result = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    return result[:top_k]


def retrieve_candidates(
    query: str,
    bm25,
    embedding_retriever,
    candidate_k: int = 100,
    bm25_k: int = 50,
    embedding_k: int = 50,
):
    bm25_results = bm25.retrieve(query, top_k=bm25_k)

    embedding_results = embedding_retriever.retrieve(
        query,
        top_k=embedding_k,
    )

    rrf_results = reciprocal_rank_fusion(
        [bm25_results, embedding_results],
        top_k=candidate_k,
    )

    return {
        "bm25": bm25_results,
        "bge": embedding_results,
        "rrf": rrf_results,
    }