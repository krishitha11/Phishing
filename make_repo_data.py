"""Make a small 'data' folder that fits on GitHub (files under 25 MB).

RoBERTa only reads the first ~256 tokens of an email, so each email is cut to
the first 1500 characters. Test sets are sampled to 3000 rows using the SAME
rows in every file (clean + modified), so comparisons stay fair. The SVM model
is scored on exactly these files too, so RoBERTa vs SVM is apples to apples.

Usage (from the phishing folder, after steps 1-3):
    python3 make_repo_data.py
Then upload the new 'data' folder to GitHub.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

SEED = 42
ALIGNED = ["test.csv", "test_typos.csv", "test_synonyms.csv",
           "test_char_swaps.csv", "test_all_combined.csv"]  # same rows, same order


def cut(df, maxchars):
    df = df.copy()
    df["text"] = df["text"].astype(str).str.slice(0, maxchars)
    return df[["text", "label"]]


def score(name, model, df):
    pred = model.predict(df["text"])
    y = df["label"].values
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(test_set=name, accuracy=accuracy_score(y, pred), precision=p, recall=r,
                f1=f1, false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0, rows=len(df))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--dest", default="data")
    ap.add_argument("--n-train", type=int, default=20000)
    ap.add_argument("--n-val", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=3000)
    ap.add_argument("--maxchars", type=int, default=1500)
    a = ap.parse_args()
    src, dst = Path(a.outputs), Path(a.dest)
    dst.mkdir(exist_ok=True)

    train = cut(pd.read_csv(src / "train.csv"), a.maxchars)
    cut_and_save = lambda df, n, name: cut(df, a.maxchars).sample(
        min(n, len(df)), random_state=SEED).to_csv(dst / name, index=False)
    train.sample(min(a.n_train, len(train)), random_state=SEED).to_csv(dst / "train.csv", index=False)
    cut_and_save(pd.read_csv(src / "val.csv"), a.n_val, "val.csv")

    base = pd.read_csv(src / "test.csv")
    idx = base.sample(min(a.n_test, len(base)), random_state=SEED).index
    sets = {}
    for f in ALIGNED:
        d = pd.read_csv(src / f)
        if len(d) != len(base):
            raise SystemExit(f"{f} has a different number of rows than test.csv")
        sets[f] = cut(d.loc[idx], a.maxchars)
    ext = pd.read_csv(src / "test_external.csv")
    sets["test_external.csv"] = cut(ext.sample(min(a.n_test, len(ext)), random_state=SEED), a.maxchars)

    model = joblib.load(src / "tfidf_svm.joblib")
    rows = []
    for f, d in sets.items():
        d.to_csv(dst / f, index=False)
        rows.append(score(f.replace("test_", "").replace("test.csv", "clean").replace(".csv", ""), model, d))
    pd.DataFrame(rows).to_csv(dst / "svm_results_same_rows.csv", index=False)

    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print()
    for p in sorted(dst.glob("*.csv")):
        mb = p.stat().st_size / 1e6
        print(f"{p.name:28s} {mb:6.1f} MB", "  <-- TOO BIG for browser upload" if mb > 24 else "")


if __name__ == "__main__":
    main()
