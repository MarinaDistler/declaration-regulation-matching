import math

import pandas as pd


def validate_predictions(
    predictions: pd.DataFrame,
    declarations: pd.DataFrame,
    regulations: pd.DataFrame,
):
    required_columns = {
        "declaration_id",
        "rank",
        "regulation_id",
        "score",
    }

    if set(predictions.columns) != required_columns:
        raise ValueError(
            f"Wrong columns: {predictions.columns.tolist()}"
        )

    declaration_ids = set(
        declarations["declaration_id"]
    )

    regulation_ids = set(
        regulations["regulation_id"]
    )

    # Все declaration_id должны быть известны.
    if not set(predictions["declaration_id"]).issubset(
        declaration_ids
    ):
        raise ValueError("Unknown declaration_id")

    # Все regulation_id должны быть известны.
    if not set(predictions["regulation_id"]).issubset(
        regulation_ids
    ):
        raise ValueError("Unknown regulation_id")

    expected_rows = len(declarations) * 10

    if len(predictions) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, "
            f"got {len(predictions)}"
        )

    for declaration_id, group in predictions.groupby(
        "declaration_id"
    ):
        if len(group) != 10:
            raise ValueError(
                f"{declaration_id}: expected 10 rows"
            )

        ranks = sorted(group["rank"].tolist())

        if ranks != list(range(1, 11)):
            raise ValueError(
                f"{declaration_id}: invalid ranks {ranks}"
            )

        if group["regulation_id"].nunique() != 10:
            raise ValueError(
                f"{declaration_id}: duplicate regulations"
            )

    if predictions["score"].isna().any():
        raise ValueError("NaN score")

    if not predictions["score"].map(math.isfinite).all():
        raise ValueError("Non-finite score")

    return True