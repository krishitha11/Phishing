"""Step 3: test the saved TF-IDF + SVM model on a DIFFERENT dataset.

Usage (from the phishing folder):
    python3 03_external_test.py external.csv --text-col "Email Text" --label-col "Email Type"

Emails that also appear in your Kaggle train/val/test files are removed first,
so the score reflects emails the model has never seen.
"""
import argparse
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

POSITIVE = {"phishing", "phishing email", "spam", "1", "1.0", "true"}


def clean(t):  # identical to 01_baseline.py so the text looks the same
    t = str(t).lower()
    t = re.sub(r"https?://\S+", " URL ", t)
    t = re.sub(r"\S+@\S+", " EMAIL ", t)
    return re.sub(r"\s+", " ", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--dir", default="outputs")
    a = ap.parse_args()
    out = Path(a.dir)

    model = joblib.load(out / "tfidf_svm.joblib")
    ext = pd.read_csv(a.csv)[[a.text_col, a.label_col]].dropna()
    ext.columns = ["text", "label"]
    print("External rows loaded:", len(ext))
    print(ext["label"].value_counts())

    ext["label"] = ext["label"].astype(str).str.lower().str.strip().isin(POSITIVE).astype(int)
    ext["text"] = ext["text"].map(clean)
    ext = ext[ext["text"].str.len() > 20].drop_duplicates(subset="text")

    seen = set()
    for n in ["train", "val", "test"]:
        seen |= set(pd.read_csv(out / f"{n}.csv")["text"])
    overlap = ext["text"].isin(seen)
    print(f"Removed {overlap.sum()} emails already seen in your Kaggle data")
    ext = ext[~overlap]
    print("External rows used:", len(ext), "| phishing:", int(ext["label"].sum()),
          "| legitimate:", int((ext["label"] == 0).sum()))

    pred = model.predict(ext["text"])
    y = ext["label"]
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    res = dict(test_set="external", accuracy=accuracy_score(y, pred), precision=p,
               recall=r, f1=f1,
               false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0,
               rows=len(ext))
    df = pd.DataFrame([res])
    print(df.round(4).to_string(index=False))
    df.to_csv(out / "external_results.csv", index=False)
    ext.to_csv(out / "test_external.csv", index=False)  # RoBERTa reuses this
    print("Saved external_results.csv and test_external.csv in", out.resolve())


if __name__ == "__main__":
    main()
