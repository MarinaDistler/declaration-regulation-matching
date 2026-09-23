import argparse
from pathlib import Path
import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import torch

from src.data import (
    load_declarations,
    load_regulations,
)
from src.preprocessing import (
    declaration_to_text,
    regulation_to_text,
)
from src.bm25 import BM25Retriever
from src.embeddings import EmbeddingRetriever
from src.reranker import Reranker
from src.retrieval import retrieve_candidates
from src.ranking import (
    make_final_ranking,
    validate_top_k,
)
from src.utils import validate_predictions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        default="./data",
    )
    parser.add_argument(
        "--models-dir",
        default="./models",
    )
    parser.add_argument(
        "--out",
        default="./out",
    )
    parser.add_argument(
        "--bm25-k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--embedding-k",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=10,
    )
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models_dir)
    out_dir = Path(args.out)

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")

    # --------------------------------------------------
    # DATA
    # --------------------------------------------------

    declarations = load_declarations(
        data_dir / "declarations.jsonl"
    )

    regulations = load_regulations(
        data_dir / "regulations.jsonl"
    )

    print(
        f"Declarations: {len(declarations)}"
    )

    print(
        f"Regulations: {len(regulations)}"
    )

    # --------------------------------------------------
    # TEXT
    # --------------------------------------------------

    declaration_texts = [
        declaration_to_text(row)
        for _, row in declarations.iterrows()
    ]

    regulation_texts = [
        regulation_to_text(row)
        for _, row in regulations.iterrows()
    ]

    # --------------------------------------------------
    # BM25
    # --------------------------------------------------

    print("Building BM25...")

    bm25 = BM25Retriever(
        regulation_texts
    )

    # --------------------------------------------------
    # EMBEDDINGS
    # --------------------------------------------------

    print("Loading embedding model...")

    embedding_retriever = EmbeddingRetriever(
        model_path=str(
            models_dir / "embedding"
        ),
        device=device,
        batch_size=32,
        use_e5_prefix=True,
    )

    print("Encoding regulations...")

    embedding_retriever.fit(
        regulation_texts
    )

    # --------------------------------------------------
    # RERANKER
    # --------------------------------------------------

    print("Loading reranker...")

    reranker = Reranker(
        model_path=str(
            models_dir / "reranker"
        ),
        device=device,
        max_length=512,
    )

    # --------------------------------------------------
    # RETRIEVAL + RERANKING
    # --------------------------------------------------

    results = []

    for i, (_, declaration) in enumerate(
        declarations.iterrows()
    ):
        declaration_id = declaration[
            "declaration_id"
        ]

        query = declaration_texts[i]

        # 1. BM25 + embeddings
        fused_candidates = retrieve_candidates(
            query=query,
            bm25=bm25,
            embedding_retriever=embedding_retriever,
            candidate_k=args.candidate_k,
            bm25_k=args.bm25_k,
            embedding_k=args.embedding_k,
        )

        candidate_indices = [
            idx
            for idx, _ in fused_candidates
        ]

        # 2. Reranker
        ranked = make_final_ranking(
            declaration_text=query,
            candidate_indices=candidate_indices,
            regulation_texts=regulation_texts,
            reranker=reranker,
            top_k=args.final_k,
        )

        # Safety check
        validate_top_k(
            ranked,
            top_k=args.final_k,
        )

        # 3. Save
        for rank, (reg_idx, score) in enumerate(
            ranked,
            start=1,
        ):
            results.append(
                {
                    "declaration_id": declaration_id,
                    "rank": rank,
                    "regulation_id": regulations.iloc[
                        reg_idx
                    ]["regulation_id"],
                    "score": float(score),
                }
            )

        if (i + 1) % 10 == 0:
            print(
                f"Processed {i + 1}/"
                f"{len(declarations)}"
            )

    # --------------------------------------------------
    # OUTPUT
    # --------------------------------------------------

    predictions = pd.DataFrame(results)

    predictions = predictions[
        [
            "declaration_id",
            "rank",
            "regulation_id",
            "score",
        ]
    ]

    validate_predictions(
        predictions,
        declarations,
        regulations,
    )

    output_path = (
        out_dir / "predictions.csv"
    )

    predictions.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Done!")
    print(f"Saved to: {output_path}")
    print(
        f"Rows: {len(predictions)}"
    )


if __name__ == "__main__":
    main()