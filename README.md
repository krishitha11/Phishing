# Explainable Phishing Email Detection

Atria Institute of Technology, Dept. of CSE. Review 1 project.
Compares a TF-IDF + SVM baseline with a fine-tuned RoBERTa model, tests both on
modified emails and on other data sources, and (planned) explains predictions with LIME.

## Data
* Main dataset: Kaggle "Phishing Email Dataset" (Naser Abdullah Alam), ~82,000 emails from
  Enron, Ling, CEAS-08, Nazario, Nigerian Fraud and SpamAssassin. Check the licence before reuse.
* External check: Hugging Face `zefang-liu/phishing-email-dataset` (LGPL-3.0). Part of its
  content comes from the same public corpora, so it is not fully independent.
* The `data/` folder holds a small version (emails cut to 1,500 characters, 3,000 rows per test set)
  so it fits on GitHub. Full files are not included.

## Method
1. Remove duplicates, then split 70/15/15 **before** any modification (no leakage).
2. TF-IDF + linear SVM baseline.
3. Robustness: typos, synonym swaps and character swaps applied to a copy of the test set only.
4. External test on the Hugging Face dataset (emails already seen are removed).
5. Leave-one-source-out: train on five sources, test on the sixth.
6. RoBERTa (`roberta-base`) fine-tuned on Colab, tested on the same files.

## Run order
| File | What it does |
|---|---|
| `01_baseline.py` | clean, deduplicate, split, SVM baseline |
| `02_robustness.py` | modified test sets and scores (`--rate 0.4` for stronger edits) |
| `03_external_test.py` | test on another dataset |
| `04_leave_one_source_out.py` | generalization across sources |
| `make_repo_data.py` | builds the small `data/` folder |
| `05_roberta_colab.ipynb` | RoBERTa fine-tuning (Colab, T4 GPU) |

## Results so far (TF-IDF + SVM, full 12,305-email test set)
| Test set | Accuracy | False-positive rate |
|---|---|---|
| Clean | 0.9935 | 0.0077 |
| Typos (40% of words) | 0.9872 | 0.0187 |
| Synonyms (40%) | 0.9934 | 0.0077 |
| Character swaps (40%) | 0.9852 | 0.0258 |
| All combined | 0.9839 | 0.0270 |
| External (Hugging Face) | 0.9936 | 0.0042 |

High clean accuracy may partly reflect source style (e.g. Enron vs Nazario), so the
leave-one-source-out test is the stronger generalization check.
RoBERTa and leave-one-source-out results: to be added.

## Limits
Controlled offline tests only. No claim of production-level security.
