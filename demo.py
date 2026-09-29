"""Interactive demo with the baseline model. Run: python src/demo.py"""
import joblib

from utils import LABELS, MODELS, build_row, prepare

model = joblib.load(MODELS / "baseline.joblib")
print("Aspect-based sentiment demo (empty sentence to quit)")
while True:
    sentence = input("\nReview sentence: ").strip()
    if not sentence:
        break
    aspect = input("Aspect word/phrase inside it: ").strip()
    try:
        row = prepare(build_row(sentence, aspect))
    except ValueError as err:
        print(err)
        continue
    probs = model.predict_proba(row)[0]
    best = probs.argmax()
    print(f"-> {LABELS[best].upper()} ({probs[best]:.0%}) about '{aspect}'")
    for label, p in sorted(zip(LABELS, probs), key=lambda x: -x[1]):
        print(f"   {label:<9}{'#' * int(p * 30):<30} {p:.0%}")
