import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        batch_size: int = 32,
        use_e5_prefix: bool = True,
    ):
        self.model = SentenceTransformer(
            model_path,
            device=device,
            local_files_only=True,
        )

        self.batch_size = batch_size
        self.use_e5_prefix = use_e5_prefix

        self.document_embeddings = None

    def _prepare_query(self, text: str) -> str:
        if self.use_e5_prefix:
            return f"query: {text}"

        return text

    def _prepare_document(self, text: str) -> str:
        if self.use_e5_prefix:
            return f"passage: {text}"

        return text

    def fit(self, documents: list[str]):
        documents = [
            self._prepare_document(doc)
            for doc in documents
        ]

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

        query = self._prepare_query(query)

        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )[0]

        # При нормализованных embedding cosine similarity
        # превращается в обычное скалярное произведение.
        scores = self.document_embeddings @ query_embedding

        top_k = min(top_k, len(scores))

        indices = np.argsort(scores)[::-1][:top_k]

        return [
            (int(idx), float(scores[idx]))
            for idx in indices
        ]