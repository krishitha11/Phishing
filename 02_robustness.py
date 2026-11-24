"""Step 2: robustness test.

Makes modified copies of outputs/test.csv (typos, synonym swaps, character
swaps), runs the saved TF-IDF + SVM model on each, and prints how much the
scores drop compared with the clean test set.

Usage (from the phishing folder, after 01_baseline.py):
    python3 02_robustness.py
"""
import argparse
import random
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)

SEED = 42

# Small hand-made synonym list (no downloads needed). Add more pairs to make
# the test stronger. Each word maps to a list of replacements.
SYNONYMS = {
    "urgent": ["immediate", "pressing"],
    "immediately": ["right away", "promptly"],
    "verify": ["confirm", "validate"],
    "account": ["profile", "login"],
    "click": ["press", "tap"],
    "free": ["complimentary", "gratis"],
    "money": ["cash", "funds"],
    "bank": ["lender", "financial institution"],
    "password": ["passcode", "login key"],
    "update": ["refresh", "renew"],
    "important": ["vital", "critical"],
    "offer": ["deal", "promotion"],
    "winner": ["victor", "awardee"],
    "prize": ["reward", "award"],
    "send": ["forward", "transmit"],
    "payment": ["remittance", "transfer"],
    "security": ["protection", "safety"],
    "attached": ["enclosed", "included"],
    "meeting": ["session", "call"],
    "please": ["kindly"],
    "thanks": ["thank you", "cheers"],
    "message": ["note", "mail"],
    "customer": ["client", "user"],
    "confirm": ["verify", "validate"],
    "link": ["url", "address"],
}

CHAR_SWAPS = {"o": "0", "i": "1", "e": "3", "a": "@", "s": "$"}


def typo(word, rng):
    """Make one small spelling mistake in a word."""
    if len(word) < 4:
        return word
    i = rng.randrange(1, len(word) - 1)
    kind = rng.choice(["swap", "drop", "double"])
    if kind == "swap":  # swap two neighbouring letters
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if kind == "drop":  # delete a letter
        return word[:i] + word[i + 1:]
    return word[:i] + word[i] + word[i:]  # repeat a letter


def char_swap(word, rng):
    """Replace one letter with a look-alike character (o -> 0 etc.)."""
    spots = [i for i, c in enumerate(word) if c in CHAR_SWAPS]
    if not spots:
        return word
    i = rng.choice(spots)
    return word[:i] + CHAR_SWAPS[word[i]] + word[i + 1:]


def synonym(word, rng):
    return rng.choice(SYNONYMS[word]) if word in SYNONYMS else word


def perturb(text, fn, rate, seed_offset=0):
    rng = random.Random(SEED + seed_offset)
    words = str(text).split()
    return " ".join(fn(w, rng) if rng.random() < rate else w for w in words)


def score(name, model, df):
    pred = model.predict(df["text"])
    y = df["label"]
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(test_set=name, accuracy=accuracy_score(y, pred), precision=p,
                recall=r, f1=f1,
                false_positive_rate=fp / (fp + tn) if (fp + tn) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs")
    ap.add_argument("--rate", type=float, default=0.15,
                    help="fraction of words to change (default 0.15)")
    a = ap.parse_args()
    out = Path(a.dir)

    model = joblib.load(out / "tfidf_svm.joblib")
    test = pd.read_csv(out / "test.csv")  # original test file is never changed

    def make(fn):
        d = test.copy()
        d["text"] = [perturb(t, fn, a.rate, i) for i, t in enumerate(d["text"])]
        return d

    def all_three(w, rng):
        return char_swap(typo(synonym(w, rng), rng), rng)

    sets = {
        "clean": test,
        "typos": make(typo),
        "synonyms": make(synonym),
        "char_swaps": make(char_swap),
        "all_combined": make(all_three),
    }

    rows = []
    for name, d in sets.items():
        if name != "clean":
            d.to_csv(out / f"test_{name}.csv", index=False)  # separate files
        rows.append(score(name, model, d))

    res = pd.DataFrame(rows)
    base = res.loc[0, "accuracy"]
    res["accuracy_drop"] = base - res["accuracy"]
    res.to_csv(out / "robustness_results.csv", index=False)
    print(res.round(4).to_string(index=False))
    print("\nSaved: robustness_results.csv and test_*.csv in", out.resolve())


if __name__ == "__main__":
    main()
