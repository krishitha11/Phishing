# WhatsApp Message Spam Detection

A prototype for screening WhatsApp messages that contain phishing-style email content or suspicious links.

## What the current model does

The Streamlit app runs a fine-tuned RoBERTa classifier and shows a LIME explanation. The checkpoint was trained and evaluated on phishing-email data. It accepts WhatsApp-style text, but WhatsApp-specific accuracy has not been measured. URLs are treated as text; the app does not open links or check domain reputation. Do not use this prototype as a security decision on its own.

## Run the app

```bash
git clone https://github.com/krishitha11/Phishing.git
cd Phishing
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On the first analysis, the app downloads the published checkpoint from the GitHub release into `models/roberta-phishing/` (about 450 MB). Later runs use the local copy. If hosted, submitted message text is processed by the hosting server; avoid entering private messages.

## Train the checkpoint

Open `05_roberta_colab.ipynb` in Google Colab and run all cells.

- With a GPU, it runs the full email-data training and evaluation and downloads a results CSV and model archive.
- Without a GPU, it runs a 128-email CPU demo, exports a checkpoint, and skips evaluation. That demo model is only for trying the interface and has no meaningful reported metrics.

## Results

The committed CSVs contain scores on six **email** test sets with 3,000 rows each. The SVM results use the same rows as the RoBERTa results. These numbers do not measure WhatsApp-message performance.

| Test set | RoBERTa accuracy | SVM accuracy | RoBERTa false-positive rate | SVM false-positive rate |
|---|---:|---:|---:|---:|
| Clean | 98.77% | 99.47% | 1.57% | 0.82% |
| Typos | 97.97% | 98.80% | 3.47% | 1.98% |
| Synonyms | 98.83% | 99.43% | 1.57% | 0.89% |
| Character swaps | 97.83% | 98.57% | 3.13% | 2.79% |
| All combined | 97.20% | 98.17% | 4.36% | 3.20% |
| External | 96.80% | 98.93% | 2.32% | 1.00% |

On these email test sets, the TF-IDF + SVM baseline outperformed RoBERTa on accuracy and false-positive rate. False-positive rate is the share of legitimate messages classified as phishing.

The Review 1 presentation also reports a separate SMS spam result. Its dataset and implementation are not included in this repository, so those SMS metrics are not mixed into the email results above.

## Files

- `app.py` — local Streamlit app with RoBERTa predictions and LIME explanations.
- `05_roberta_colab.ipynb` — checkpoint training and evaluation.
- `01_baseline.py` through `04_leave_one_source_out.py` — email dataset baseline and robustness scripts.
- `data/` — small email dataset samples and evaluation CSVs.
- `requirements.txt` — app dependencies.

The model weights are distributed as a GitHub release asset rather than stored in Git history. The repository ignores the downloaded `models/roberta-phishing/` folder.
