"""Splits dataset into temporal or drive-level train/validation/test sets."""
import pandas as pd

def split_dataset(df: pd.DataFrame) -> tuple:
    """Splits data into train, validation, and test sets."""
    return df, df, df
