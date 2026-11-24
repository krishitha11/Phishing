# WhatsApp Spam Detection: Phishing Content and Links

A machine-learning project aimed at identifying spam shared through WhatsApp, with a focus on phishing-style email content and suspicious links embedded in messages. The initial baseline described in Review 1 classifies SMS messages as spam or legitimate; WhatsApp-specific data and evaluation are still needed before claiming performance on WhatsApp messages.

This project detects spam-like messages. It does not verify whether a message's factual claims are true.

## Review 1 baseline

The presentation describes a local message-classification app using:

- The UCI SMS Spam Collection, with 5,159 unique messages after exact-duplicate removal.
- TF-IDF text features and a calibrated linear SVM.
- A prediction, confidence score, and influential terms for each message.
- A stratified holdout result reported as 98.55% accuracy, 0.95 spam precision, and 0.93 spam recall.

These metrics are reported in the presentation and are SMS results. They have not been reproduced from the code currently in this repository, and they should not be presented as WhatsApp-specific results.

## Repository status

The files currently checked into this repository are from an earlier phishing-email project. For example, `data/` contains email datasets, `01_baseline.py` trains an email classifier, and `app.py` expects a fine-tuned RoBERTa email model with LIME explanations. They do not implement the SMS baseline described above or a WhatsApp spam detector.

The next implementation step is to add the SMS spam dataset, baseline model, and message interface described in Review 1, then evaluate on WhatsApp-specific messages, especially messages containing suspicious links and phishing-style content, before reporting WhatsApp performance.

## Intended scope

- Classify spam and legitimate messages shared through WhatsApp, especially phishing-style content and suspicious links.
- Start with an SMS spam baseline and explain the terms that influence its prediction.
- Evaluate on WhatsApp-specific messages containing suspicious links before making WhatsApp performance claims.

This is a research prototype, not a guarantee that a message is safe or harmful.
