"""Step 4: leave-one-source-out test.

The Kaggle dataset is built from several sources (CEAS_08, Enron, Ling,
Nazario, Nigerian_Fraud, SpamAssassin). For each source in turn: train
TF-IDF + SVM on all the OTHER sources, then test on the source it never saw.

Usage:
    python3 04_leave_one_source_out.py PATH_TO_UNZIPPED_KAGGLE_FOLDER
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

SEED = 42
POSITIVE = {"1", "1.0", "phishing", "phishing email", "spam", "true"}
SKIP = {"phishing_email.csv"}  # the combined file; we use the separate sources


def clean(t):  # same cleaning as 01_baseline.py
    t = str(t).lower()
    t = re.sub(r"https?://\S+", " URL ", t)
    t = re.sub(r"\S+@\S+", " EMAIL ", t)
    return re.sub(r"\s+", " ", t).strip()


def load_source(path):
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    if "label" not in cols or not ({"body", "subject"} & set(cols)):
        raise SystemExit(f"{path.name}: unexpected columns {list(df.columns)}. "
                         "Need 'label' and 'subject'/'body'.")
    parts = [df[cols[c]].fillna("").astype(str) for c in ("subject", "body") if c in cols]
    text = parts[0] if len(parts) == 1 else parts[0] + " " + parts[1]
    out = pd.DataFrame({"text": text.map(clean),
                        "label": df[cols["label"]].astype(str).str.lower().str.strip()
                                   .isin(POSITIVE).astype(int)})
    out["source"] = path.stem
    return out[out["text"].str.len() > 20]


def metrics(y, pred):
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else np.nan  # n/a if no legitimate emails
    return accuracy_score(y, pred), p, r, f1, fpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    files = sorted(f for f in Path(a.folder).expanduser().glob("*.csv")
                   if f.name.lower() not in SKIP)
    if len(files) < 3:
        raise SystemExit(f"Found only {len(files)} source CSVs in {a.folder}")
    data = pd.concat([load_source(f) for f in files], ignore_index=True)
    before = len(data)
    data = data.drop_duplicates(subset="text")  # duplicates across sources would leak
    print(f"Loaded {before} emails, {len(data)} after removing duplicates")
    print(data.groupby("source")["label"].agg(["count", "mean"]).rename(
        columns={"mean": "share_phishing"}).round(2))

    rows = []
    for src in sorted(data["source"].unique()):
        train, test = data[data["source"] != src], data[data["source"] == src]
        if len(test) == 0 or train["label"].nunique() < 2:
            continue
        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True),
            LinearSVC(C=1.0, class_weight="balanced", random_state=SEED))
        model.fit(train["text"], train["label"])
        acc, p, r, f1, fpr = metrics(test["label"], model.predict(test["text"]))
        rows.append(dict(held_out_source=src, test_rows=len(test),
                         phishing=int(test["label"].sum()),
                         legitimate=int((test["label"] == 0).sum()),
                         accuracy=acc, precision=p, recall=r, f1=f1,
                         false_positive_rate=fpr))
        print(f"done: {src}")

    res = pd.DataFrame(rows)
    Path(a.out).mkdir(exist_ok=True)
    res.to_csv(Path(a.out) / "leave_one_source_out.csv", index=False)
    print("\n", res.round(4).to_string(index=False))
    print("\nFor sources with 0 legitimate emails only 'recall' is meaningful.")
    print("Saved outputs/leave_one_source_out.csv")


if __name__ == "__main__":
    main()
