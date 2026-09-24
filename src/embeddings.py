import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 32,
    ):
        self.model = SentenceTransformer(
            model_path,
            device=device,
            local_files_only=True,
        )

        self.batch_size = batch_size
        self.document_embeddings = None

    def fit(self, documents: list[str]):
        self.document_embeddings = self.model.encode(
            documents,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
    ) -> list[tuple[int, float]]:

        if self.document_embeddings is None:
            raise RuntimeError("Call fit() before retrieve().")

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # При нормализованных embedding:
        # cosine similarity = скалярное произведение.
        scores = self.document_embeddings @ query_embedding

        top_k = min(top_k, len(scores))
        indices = np.argsort(scores)[::-1][:top_k]

        return [
            (int(idx), float(scores[idx]))
            for idx in indices
        ]