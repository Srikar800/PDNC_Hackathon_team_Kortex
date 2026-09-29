"""Quick exploratory data analysis. Run: python src/eda.py"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import OUT, load_data

train, test, sample = load_data()
print("train", train.shape, "| test", test.shape, "| sample_submission", sample.shape)
print("\nMissing values (train):\n", train.isna().sum().to_string())
print("\nLabel distribution:\n", train["label"].value_counts().to_string())
print("\nSentence length (words):\n", train["text"].str.split().str.len().describe().round(1).to_string())
print("\nUnique sentences:", train["text"].nunique(), "of", len(train))
print("Test sentences also present in train:", round(test["text"].isin(train["text"]).mean() * 100, 1), "%")
print("Test ids match sample_submission ids:", (test["id"].values == sample["id"].values).all())

fig, ax = plt.subplots(figsize=(5, 3.5))
train["label"].value_counts().plot.bar(ax=ax, color=["#8a8a8a", "#5cb85c", "#d9534f"])
ax.set_title("Class distribution (train)")
ax.set_ylabel("rows")
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=150)
print("\nSaved", OUT / "class_distribution.png")
