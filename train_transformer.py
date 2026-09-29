"""Fine-tune a transformer for aspect-based sentiment.

Run (Colab GPU recommended):  python src/train_transformer.py
Optional:                     python src/train_transformer.py --model distilroberta-base --epochs 4

Input to the model is a sentence PAIR:  [aspect term]  [full sentence]
so the model knows which aspect to judge. Plain PyTorch loop (no Trainer)
to avoid version-specific argument names.
"""
import argparse

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from utils import (ID2LABEL, LABEL2ID, LABELS, MODELS, OUT, ROOT, SEED,
                   grouped_split, load_data, save_val_predictions, write_submission)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="roberta-base")
parser.add_argument("--epochs", type=int, default=4)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--max_len", type=int, default=128)
args = parser.parse_args()

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device, "| model:", args.model)

train, test, sample = load_data()
tr, val = grouped_split(train)          # same grouped split as the baseline
tokenizer = AutoTokenizer.from_pretrained(args.model)


def make_loader(df, shuffle, labels=True):
    """Tokenize (aspect, sentence) pairs and wrap them in a DataLoader."""
    enc = tokenizer(list(df["aspect_term"]), list(df["text"]), truncation=True,
                    max_length=args.max_len, padding=False)
    items = []
    for i in range(len(df)):
        item = {k: enc[k][i] for k in enc.keys()}
        if labels:
            item["labels"] = LABEL2ID[df["label"].iloc[i]]
        items.append(item)
    collate = lambda batch: tokenizer.pad(batch, return_tensors="pt")  # dynamic padding
    return DataLoader(items, batch_size=args.batch_size, shuffle=shuffle, collate_fn=collate)


def predict(model, loader):
    """Return class probabilities for every row in the loader."""
    model.eval()
    out = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            out.append(torch.softmax(model(**batch).logits, dim=-1).cpu().numpy())
    return np.concatenate(out)


train_loader = make_loader(tr, shuffle=True)
val_loader = make_loader(val, shuffle=False)
test_loader = make_loader(test, shuffle=False, labels=False)

model = AutoModelForSequenceClassification.from_pretrained(
    args.model, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
total_steps = len(train_loader) * args.epochs
scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

y_val = val["label"].map(LABEL2ID).values
best_f1, best_val_probs, best_test_probs = -1.0, None, None

for epoch in range(1, args.epochs + 1):
    model.train()
    running = 0.0
    for step, batch in enumerate(train_loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        running += loss.item()
    val_probs = predict(model, val_loader)
    val_pred = val_probs.argmax(axis=1)
    acc, f1 = accuracy_score(y_val, val_pred), f1_score(y_val, val_pred, average="macro")
    print(f"epoch {epoch}/{args.epochs} | train loss {running / len(train_loader):.4f} "
          f"| val acc {acc:.4f} | val macro-F1 {f1:.4f}")
    if f1 > best_f1:                      # keep the best epoch (early-stopping style)
        best_f1, best_val_probs = f1, val_probs
        best_test_probs = predict(model, test_loader)
        model.save_pretrained(MODELS / "transformer")
        tokenizer.save_pretrained(MODELS / "transformer")
        print("  -> new best, saved")

print(f"best validation macro-F1: {best_f1:.4f}")
save_val_predictions("transformer", val, best_val_probs.argmax(axis=1), best_val_probs)
write_submission(test, sample, [LABELS[i] for i in best_test_probs.argmax(axis=1)],
                 OUT / "submission_transformer.csv")
print("saved:", OUT / "submission_transformer.csv")
print("If its validation macro-F1 beats the baseline, copy it over submission.csv")
