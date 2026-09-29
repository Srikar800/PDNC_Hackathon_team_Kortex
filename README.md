# Aspect-Based Sentiment Analysis (restaurant reviews)

Given a review sentence and one aspect in it (e.g. "waiter", "food"), predict the
sentiment **towards that aspect**: `Negative`, `Neutral` or `Positive`.

Example: *"The food was great but the waiter was rude."* -> food = Positive, waiter = Negative.

## Data (from the hackathon files)
- `train.csv`: 8,000 rows | `test.csv`: 2,000 rows | `sample_submission.csv`: `id,prediction`
- Classes: Neutral 45%, Positive 30%, Negative 25%
- ~79% of test sentences also appear in train (with a different aspect), so validation
  is split **by sentence** to avoid inflated scores.

## Setup (Windows / Linux / Mac)
```
pip install -r requirements.txt
python src/eda.py
python src/train_baseline.py      # trains, writes submission.csv
python src/evaluate.py baseline   # accuracy, F1, confusion matrix
python src/demo.py                # type a sentence + aspect
```

Optional stronger model (needs `pip install torch transformers`, GPU recommended, e.g. Google Colab):
```
python src/train_transformer.py
python src/evaluate.py transformer
```
This writes `outputs/submission_transformer.csv`. Use it as `submission.csv` only if its
validation macro-F1 is higher than the baseline's.

## Method
- **Baseline:** TF-IDF on (1) the sentence, (2) the words next to the aspect tagged left/right,
  (3) the aspect word; Logistic Regression with balanced class weights.
- **Transformer:** RoBERTa fine-tuned on the pair (aspect, sentence).

## Results (validation, sentence-grouped 20% hold-out)
| Model | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.7086 | 0.6981 |
| Transformer | not run yet | not run yet |

## Limitations
Trained on restaurant reviews only; it will not transfer well to other domains.
Predictions reflect patterns in this dataset and can be wrong on sarcasm or mixed opinions.
