import numpy as np
from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.model = CrossEncoder(
            model_path,
            device=device,
            max_length=max_length,
            local_files_only=True,
        )

    def rerank(
        self,
        query: str,
        candidates: list[tuple[int, str]],
        top_k: int = 10,
    ) -> list[tuple[int, float]]:
        """
        candidates:
            [(document_index, document_text), ...]

        returns:
            [(document_index, reranker_score), ...]
        """

        if not candidates:
            return []

        pairs = [
            (query, document_text)
            for _, document_text in candidates
        ]

        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False,
        )

        scores = np.asarray(scores)

        order = np.argsort(scores)[::-1][:top_k]

        return [
            (
                candidates[int(i)][0],
                float(scores[int(i)]),
            )
            for i in order
        ]