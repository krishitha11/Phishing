"""Step 1: clean data -> split BEFORE augmentation -> TF-IDF + SVM baseline.

Usage:
    python 01_baseline.py data.csv --text-col text --label-col label

The CSV needs an email-body column and a label column. Labels can be
0/1 or words like "phishing"/"legitimate"; edit POSITIVE below if needed.
"""
import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

POSITIVE = {"phishing", "phishing email", "spam", "1", "1.0", "true"}
SEED = 42


def clean(t: str) -> str:
    t = str(t).lower()
    t = re.sub(r"https?://\S+", " URL ", t)
    t = re.sub(r"\S+@\S+", " EMAIL ", t)
    return re.sub(r"\s+", " ", t).strip()


def evaluate(name, model, X, y):
    pred = model.predict(X)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    res = dict(set=name, accuracy=accuracy_score(y, pred), precision=p, recall=r,
               f1=f1, false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    df = pd.read_csv(a.csv)[[a.text_col, a.label_col]].dropna()
    df.columns = ["text", "label"]
    print("Rows loaded:", len(df))
    print(df["label"].value_counts())

    df["label"] = df["label"].astype(str).str.lower().str.strip().isin(POSITIVE).astype(int)
    df["text"] = df["text"].map(clean)
    df = df[df["text"].str.len() > 20]
    df = df.drop_duplicates(subset="text")  # exact duplicates only; near-dups come later
    print("Rows after cleaning + dedup:", len(df))

    # Split FIRST. Any augmentation later touches the training set only.
    train, temp = train_test_split(df, test_size=0.30, stratify=df["label"], random_state=SEED)
    val, test = train_test_split(temp, test_size=0.50, stratify=temp["label"], random_state=SEED)
    for n, d in [("train", train), ("val", val), ("test", test)]:
        d.to_csv(out / f"{n}.csv", index=False)  # fixed files reused by RoBERTa later
    print({n: len(d) for n, d in [("train", train), ("val", val), ("test", test)]})

    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
        LinearSVC(C=1.0, class_weight="balanced", random_state=SEED),
    )
    model.fit(train["text"], train["label"])

    results = [evaluate("val", model, val["text"], val["label"]),
               evaluate("test", model, test["text"], test["label"])]
    (out / "baseline_metrics.json").write_text(json.dumps(results, indent=2))
    joblib.dump(model, out / "tfidf_svm.joblib")
    print("Saved to", out.resolve())


if __name__ == "__main__":
    main()
