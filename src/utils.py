from typing import List

import pandas as pd


# Competition metric function
def CompetitionMetric(
    df: pd.DataFrame,
    models: List[str],
    id_col: str = "unique_id",
    target_col: str = "y",
) -> pd.DataFrame:
    """Computes the competition metric"""
    error = df[models].sub(df[target_col], axis=0)
    sum_abs_error = error.abs().sum()
    abs_sum_error = error.sum().abs()
    score = sum_abs_error + abs_sum_error
    score /= df[target_col].sum()
    score.index.name = id_col
    score = score.reset_index()
    score.columns = [id_col] + models
    return score
