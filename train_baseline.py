"""Fast classical baseline: TF-IDF + Logistic Regression. Run: python src/train_baseline.py

Features
  - sentence   : word 1-2 grams of the whole sentence (general tone)
  - window     : words right next to the aspect, tagged L_/R_ (aspect-specific tone)
  - aspect     : the aspect word itself (e.g. 'waiter' is often negative in this data)
"""
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from utils import (LABEL2ID, LABELS, MODELS, ROOT, grouped_split, load_data,
                   prepare, save_val_predictions, write_submission)


def make_model():
    features = ColumnTransformer([
        ("sentence", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True), "text"),
        ("window", TfidfVectorizer(token_pattern=r"\S+", ngram_range=(1, 2), sublinear_tf=True), "window"),
        ("aspect", TfidfVectorizer(token_pattern=r"\S+"), "aspect"),
    ])
    clf = LogisticRegression(C=3.0, max_iter=2000, class_weight="balanced")
    return Pipeline([("features", features), ("clf", clf)])


if __name__ == "__main__":
    train, test, sample = load_data()
    train, test = prepare(train), prepare(test)

    # 1) Validate on a sentence-grouped hold-out
    tr, val = grouped_split(train)
    print(f"train part: {len(tr)} rows | validation part: {len(val)} rows")
    print("sentence overlap between them:", len(set(tr['text']) & set(val['text'])), "(must be 0)")

    model = make_model()
    model.fit(tr, tr["label"].map(LABEL2ID))
    probs = model.predict_proba(val)
    pred = probs.argmax(axis=1)
    y_val = val["label"].map(LABEL2ID)
    print(f"validation accuracy: {accuracy_score(y_val, pred):.4f}")
    print(f"validation macro-F1: {f1_score(y_val, pred, average='macro'):.4f}")
    print("saved:", save_val_predictions("baseline", val, pred, probs))

    # 2) Refit on ALL training rows and predict the test set
    final = make_model()
    final.fit(train, train["label"].map(LABEL2ID))
    joblib.dump(final, MODELS / "baseline.joblib")
    test_pred = final.predict(test)
    sub = write_submission(test, sample, [LABELS[i] for i in test_pred], ROOT / "submission.csv")
    print("saved:", ROOT / "submission.csv")
    print(sub["prediction"].value_counts().to_string())
