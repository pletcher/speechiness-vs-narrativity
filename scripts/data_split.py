from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

LABELS = {"tragedy": 0, "epic": 1}
LABEL2ID = LABELS
ID2LABEL = {i: label for label, i in LABELS.items()}


def load_and_split(csv_path: Path, test_size: float = 0.2, random_state: int = 42):
    """Load the sentence CSV and split into train/eval/test.

    Splits on unique sentence text before assigning rows, so duplicate/repeated
    sentences (e.g. Homeric formulae) can't leak from train into eval/test.
    """
    df = pd.read_csv(csv_path)
    df["label"] = df.register.map(LABELS)
    assert df["label"].notna().all(), (
        "found register value(s) not in {'tragedy', 'epic'}"
    )

    unique_texts = df.drop_duplicates(subset="text")[["text", "label"]]
    train_texts, eval_test_texts = train_test_split(
        unique_texts,
        test_size=test_size,
        stratify=unique_texts["label"],
        random_state=random_state,
    )
    eval_texts, test_texts = train_test_split(
        eval_test_texts,
        test_size=0.5,
        stratify=eval_test_texts["label"],
        random_state=random_state,
    )

    train_df = df[df["text"].isin(train_texts["text"])]
    eval_df = df[df["text"].isin(eval_texts["text"])]
    test_df = df[df["text"].isin(test_texts["text"])]

    return train_df, eval_df, test_df
