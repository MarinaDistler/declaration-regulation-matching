import numpy as np
import pandas as pd


def precision_at_10(ranking, relevance):
    top10 = ranking[:10]

    return sum(
        relevance.get(regulation_id, 0) == 1
        for regulation_id, _ in top10
    ) / 10


def ndcg_at_10(ranking, relevance):
    top10 = ranking[:10]

    # DCG@10
    dcg = 0.0

    for rank, (regulation_id, _) in enumerate(top10, start=1):
        rel = relevance.get(regulation_id, 0)
        dcg += rel / np.log2(rank + 1)

    # IDCG@10
    n_relevant = sum(relevance.values())

    idcg = sum(
        1 / np.log2(rank + 1)
        for rank in range(1, min(n_relevant, 10) + 1)
    )

    if idcg == 0:
        return 0.0

    return dcg / idcg

