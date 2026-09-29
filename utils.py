"""Shared helpers: paths, data loading, grouped split, feature prep, metrics."""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# Paths are built from this file's location, so they work on Windows and Linux
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
MODELS.mkdir(exist_ok=True)
OUT.mkdir(exist_ok=True)

# Fixed label order -> the same id mapping everywhere (training, eval, demo)
LABELS = ["Negative", "Neutral", "Positive"]
LABEL2ID = {name: i for i, name in enumerate(LABELS)}
ID2LABEL = {i: name for name, i in LABEL2ID.items()}
SEED = 42


def load_data():
    """Load the three CSVs.

    keep_default_na=False matters: one aspect_term is literally a string that
    pandas would otherwise silently turn into NaN.
    """
    train = pd.read_csv(DATA / "train.csv", keep_default_na=False)
    test = pd.read_csv(DATA / "test.csv", keep_default_na=False)
    sample = pd.read_csv(DATA / "sample_submission.csv", keep_default_na=False)
    return train, test, sample


def grouped_split(train, n_splits=5):
    """Hold out ~20% of train for validation, grouped by sentence.

    ~79% of test sentences also occur in train (with a different aspect), so
    the same sentence appearing in both our train and validation parts would
    inflate scores. Grouping by `text` keeps every sentence on one side only.
    """
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(splitter.split(train, train["label"], groups=train["text"]))
    return train.iloc[train_idx].reset_index(drop=True), train.iloc[val_idx].reset_index(drop=True)


def local_window(context, k=4):
    """Words within k positions of the aspect, tagged with side (L_/R_).

    `context` is the sentence with the aspect replaced by $T$.
    Sentiment about an aspect is mostly decided by the words right next to it.
    """
    left, _, right = context.partition("$T$")
    left_words = re.findall(r"[a-z0-9']+", left.lower())[-k:]
    right_words = re.findall(r"[a-z0-9']+", right.lower())[:k]
    return " ".join(["L_" + w for w in left_words] + ["R_" + w for w in right_words])


def prepare(df):
    """Add the engineered text columns used by the TF-IDF model."""
    df = df.copy()
    df["window"] = df["context_with_aspect_tag"].apply(local_window)
    df["aspect"] = df["aspect_term"].str.lower()
    return df


def build_row(sentence, aspect):
    """Build one dataframe row from free text (used by the demo)."""
    start = sentence.lower().find(aspect.lower())
    if start < 0:
        raise ValueError(f'Aspect "{aspect}" was not found in the sentence.')
    end = start + len(aspect)
    return pd.DataFrame([{
        "text": sentence,
        "aspect_term": sentence[start:end],
        "context_with_aspect_tag": sentence[:start] + "$T$" + sentence[end:],
    }])


def save_val_predictions(name, val_df, pred_ids, probs):
    """Write validation predictions so evaluate.py can score any model."""
    out = pd.DataFrame({
        "id": val_df["id"],
        "true": val_df["label"],
        "pred": [ID2LABEL[int(i)] for i in pred_ids],
    })
    for i, label in enumerate(LABELS):
        out[f"p_{label}"] = probs[:, i]
    path = OUT / f"val_predictions_{name}.csv"
    out.to_csv(path, index=False)
    return path


def write_submission(test, sample, pred_labels, path):
    """Write a submission with exactly the columns/order of sample_submission.csv."""
    assert (test["id"].values == sample["id"].values).all(), "test/sample id order differs"
    sub = pd.DataFrame({"id": test["id"], "prediction": pred_labels})
    assert list(sub.columns) == list(sample.columns), "column mismatch"
    assert set(sub["prediction"]) <= set(LABELS), "unexpected label in predictions"
    assert len(sub) == len(sample), "row count mismatch"
    sub.to_csv(path, index=False)
    return sub
