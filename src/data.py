import json
from pathlib import Path

import pandas as pd


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)

    records = []

    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON at {path}, line {line_number}"
                ) from e

    return records


def load_declarations(path: str | Path) -> pd.DataFrame:
    records = load_jsonl(path)

    df = pd.DataFrame(records)

    if "declaration_id" not in df.columns:
        raise ValueError("declaration_id is missing")

    if "G31_1" not in df.columns:
        raise ValueError("G31_1 is missing")

    return df


def load_regulations(path: str | Path) -> pd.DataFrame:
    records = load_jsonl(path)

    df = pd.DataFrame(records)

    required = {"regulation_id", "decree_number", "npa"}

    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing regulation fields: {missing}")

    return df