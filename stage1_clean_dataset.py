"""
Pipeline stage 1: cleaning.

Cleans a conversation CSV so its 'Text' column is guaranteed to be plain
strings before it's fed into stage 2 (avoids the "expected string or
bytes-like object, got 'NoneType'" error that comes from non-str/non-float
values slipping past .fillna()/.astype(str)).

Usage:
    from stage1_clean_dataset import clean_dataset
    clean_dataset("conversation.csv")                  # -> dataset_fresh.csv
    clean_dataset("conversation.csv", "my_clean.csv")  # custom output name

From a terminal:
    python stage1_clean_dataset.py conversation.csv
    python stage1_clean_dataset.py conversation.csv dataset_fresh.csv
"""
import sys

import pandas as pd


def to_clean_text(x):
    """Coerce any value to a real string, never None/NaN."""
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def clean_dataset(input_csv, output_csv="dataset_fresh.csv", text_column="Text", verbose=True):
    """
    Reads `input_csv`, sanitizes `text_column` so every value is a plain
    string, and writes the result to `output_csv`.

    Also reports how many rows had a non-str/non-float value in
    `text_column`, so you can go check whether they were junk or
    something worth handling differently (e.g. a list of messages
    crammed into one cell).

    Returns the path to `output_csv`.
    """
    df = pd.read_csv(input_csv)

    # Drop leftover unnamed index columns (e.g. "Unnamed: 0") that show up
    # when a CSV was previously saved with index=True.
    unnamed_cols = [c for c in df.columns if str(c).startswith("Unnamed:")]
    if unnamed_cols:
        if verbose:
            print(f"Dropping unnamed index column(s): {unnamed_cols}")
        df = df.drop(columns=unnamed_cols)

    if text_column not in df.columns:
        raise KeyError(
            f"Column '{text_column}' not found in {input_csv}. "
            f"Available columns: {list(df.columns)}"
        )

    is_odd_type = ~df[text_column].apply(lambda x: isinstance(x, (str, float)) or x is None)
    n_odd = int(is_odd_type.sum())

    if verbose and n_odd > 0:
        print(f"Found {n_odd} row(s) with a non-str/non-float value in '{text_column}':")
        print(df.loc[is_odd_type, text_column].head(10))

    df[text_column] = df[text_column].apply(to_clean_text)
    df.to_csv(output_csv, index=False)

    if verbose:
        n_blank = int((df[text_column] == "").sum())
        print(f"Cleaned {len(df)} rows ({n_blank} now blank) -> {output_csv}")

    return output_csv


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python stage1_clean_dataset.py <input_csv> [output_csv]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "dataset_fresh.csv"
    clean_dataset(input_path, output_path)
