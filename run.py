import argparse
from pathlib import Path
import os
import random

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
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
from src.embeddings import EmbeddingRetriever
from src.ranking import validate_top_k
from src.utils import validate_predictions


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)


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
        "--embedding-k",
        type=int,
        default=10,
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
        data_dir / "declarations (4).jsonl"
    )

    regulations = load_regulations(
        data_dir / "regulations (4).jsonl"
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
    # EMBEDDINGS
    # --------------------------------------------------

    print("Loading BGE-M3...")

    embedding_retriever = EmbeddingRetriever(
        model_path=str(
            models_dir / "bge-m3"
        ),
        device=device,
        batch_size=32,
    )

    print("Encoding regulations...")

    embedding_retriever.fit(
        regulation_texts
    )

    # --------------------------------------------------
    # RETRIEVAL
    # --------------------------------------------------

    results = []

    for i, (_, declaration) in enumerate(
        declarations.iterrows()
    ):
        declaration_id = declaration[
            "declaration_id"
        ]

        query = declaration_texts[i]

        ranked = embedding_retriever.retrieve(
            query=query,
            top_k=args.embedding_k,
        )

        # Safety check
        validate_top_k(
            ranked,
            top_k=args.final_k,
        )

        # --------------------------------------------------
        # OUTPUT
        # --------------------------------------------------

        for rank, (reg_idx, score) in enumerate(
            ranked[:args.final_k],
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
    # VALIDATE OUTPUT
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