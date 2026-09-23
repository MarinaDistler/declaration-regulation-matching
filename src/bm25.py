import re

from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    """
    Простая токенизация.

    Сохраняем:
    - кириллицу
    - латиницу
    - цифры
    - дробные/технические обозначения
    """

    text = text.lower()

    return re.findall(
        r"[a-zа-яё]+(?:[-/][a-zа-яё0-9]+)*|\d+(?:[.,/-]\d+)*",
        text,
    )


class BM25Retriever:
    def __init__(self, documents: list[str]):
        self.documents = documents
        self.tokenized_documents = [
            tokenize(doc) for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_documents)

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
    ) -> list[tuple[int, float]]:
        """
        Возвращает:
            [(document_index, score), ...]
        """

        query_tokens = tokenize(query)

        scores = self.bm25.get_scores(query_tokens)

        top_k = min(top_k, len(scores))

        # argsort по убыванию
        indices = scores.argsort()[::-1][:top_k]

        return [
            (int(idx), float(scores[idx]))
            for idx in indices
        ]