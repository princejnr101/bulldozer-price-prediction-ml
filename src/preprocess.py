from __future__ import annotations

import pandas as pd


def add_dateparts(df: pd.DataFrame, date_col: str = "saledate", drop: bool = True) -> pd.DataFrame:
    """
    Extract time-based features from a date column and optionally drop the original datetime.
    Default drop=True prevents sklearn datetime dtype promotion issues.
    """
    df = df.copy()

    if date_col not in df.columns:
        return df

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    dt = df[date_col]

    df[f"{date_col}_year"] = dt.dt.year
    df[f"{date_col}_month"] = dt.dt.month
    df[f"{date_col}_day"] = dt.dt.day
    df[f"{date_col}_dayofweek"] = dt.dt.dayofweek
    df[f"{date_col}_dayofyear"] = dt.dt.dayofyear
    df[f"{date_col}_quarter"] = dt.dt.quarter
    df[f"{date_col}_is_month_start"] = dt.dt.is_month_start.astype("int8")
    df[f"{date_col}_is_month_end"] = dt.dt.is_month_end.astype("int8")
    df[f"{date_col}_is_quarter_start"] = dt.dt.is_quarter_start.astype("int8")
    df[f"{date_col}_is_quarter_end"] = dt.dt.is_quarter_end.astype("int8")

    if drop:
        df = df.drop(columns=[date_col])

    return df
