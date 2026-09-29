"""Score saved validation predictions. Run: python src/evaluate.py [baseline|transformer]"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix, f1_score)

from utils import LABELS, OUT

name = sys.argv[1] if len(sys.argv) > 1 else "baseline"
df = pd.read_csv(OUT / f"val_predictions_{name}.csv")

print(f"=== {name} ===")
print(f"accuracy : {accuracy_score(df['true'], df['pred']):.4f}")
print(f"macro-F1 : {f1_score(df['true'], df['pred'], average='macro'):.4f}")
print(classification_report(df["true"], df["pred"], labels=LABELS, digits=3))

cm = confusion_matrix(df["true"], df["pred"], labels=LABELS)
print("confusion matrix (rows = true, cols = predicted):")
print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

fig, ax = plt.subplots(figsize=(5, 4.5))
ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap="Blues", colorbar=False)
ax.set_title(f"Confusion matrix - {name}")
plt.tight_layout()
path = OUT / f"confusion_matrix_{name}.png"
plt.savefig(path, dpi=150)
print("saved:", path)
