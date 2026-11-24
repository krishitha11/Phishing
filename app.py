"""Streamlit interface for the fine-tuned RoBERTa phishing classifier."""
from pathlib import Path
import os
import shutil
import tempfile
import zipfile
from pathlib import PurePosixPath
from urllib.request import Request, urlopen

import numpy as np
import streamlit as st


DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "roberta-phishing"
MODEL_ARCHIVE_URL = (
    "https://github.com/krishitha11/Phishing/releases/download/v0.1.0/"
    "roberta-phishing-model.zip"
)
REQUIRED_MODEL_FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}


def download_checkpoint(target: Path) -> None:
    """Download and safely unpack the published checkpoint on first use."""
    if target.exists():
        raise FileNotFoundError(
            f"Checkpoint folder exists but is incomplete: {target}. "
            "Choose a complete checkpoint folder or remove the incomplete folder first."
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="whatsapp-spam-model-") as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / "model.zip"
        staging_path = temp_path / "checkpoint"
        request = Request(MODEL_ARCHIVE_URL, headers={"User-Agent": "WhatsApp-Spam-Demo"})
        with urlopen(request, timeout=90) as response, archive_path.open("wb") as out:
            shutil.copyfileobj(response, out)

        staging_path.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                relative = PurePosixPath(item.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"Unsafe model archive path: {item.filename}")
                destination = staging_path.joinpath(*relative.parts)
                if item.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(item) as source, destination.open("wb") as out:
                    shutil.copyfileobj(source, out)

        missing = REQUIRED_MODEL_FILES - {p.name for p in staging_path.iterdir() if p.is_file()}
        if missing:
            raise ValueError(f"Model archive is missing required files: {', '.join(sorted(missing))}")
        shutil.copytree(staging_path, target)


@st.cache_resource(show_spinner="Loading RoBERTa model…")
def load_model(model_dir: str):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    path = Path(model_dir).expanduser()
    if not path.is_dir():
        if path.resolve() == DEFAULT_MODEL_DIR.resolve():
            download_checkpoint(path)
        else:
            raise FileNotFoundError(f"Model folder not found: {path}")
    missing = REQUIRED_MODEL_FILES - {p.name for p in path.iterdir() if p.is_file()}
    if missing:
        raise FileNotFoundError(f"Checkpoint is missing files: {', '.join(sorted(missing))}")
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
    model.eval()
    return tokenizer, model, torch


def probability_fn(texts, tokenizer, model, torch, max_length: int):
    """Return [legitimate, phishing] probabilities for LIME."""
    results = []
    device = next(model.parameters()).device
    for start in range(0, len(texts), 16):
        batch = [str(text) for text in texts[start : start + 16]]
        encoded = tokenizer(batch, truncation=True, max_length=max_length,
                            padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            probs = torch.softmax(model(**encoded).logits, dim=-1).cpu().numpy()
        if probs.shape[1] != 2:
            raise ValueError(f"Expected a binary classifier; checkpoint has {probs.shape[1]} output labels.")
        results.extend(probs)
    return np.asarray(results)


st.set_page_config(page_title="WhatsApp Message Spam Detection", page_icon="💬", layout="centered")
st.title("WhatsApp Message Spam Detection")
st.caption("Email-trained phishing and spam text classifier with LIME explanations")

with st.sidebar:
    st.header("Model")
    model_dir = st.text_input(
        "Checkpoint folder",
        value=os.environ.get("PHISHING_MODEL_DIR", str(DEFAULT_MODEL_DIR)),
        help="A folder containing the exported Hugging Face model and tokenizer files.",
    )
    max_length = st.slider("Maximum tokens", min_value=64, max_value=512, value=256, step=32)
    num_features = st.slider("Words shown in explanation", min_value=5, max_value=30, value=12)

st.info(
    "Paste a message below. This checkpoint was trained on email data; its accuracy on "
    "WhatsApp messages has not been evaluated. Links are treated as text, not checked for safety."
)
st.caption("If the checkpoint is not already installed, the app downloads it on first use. When hosted, submitted messages are processed by the hosting server.")
email = st.text_area("Message text", height=240, placeholder="Paste a WhatsApp message here…")

if st.button("Analyze message", type="primary", disabled=not email.strip()):
    try:
        tokenizer, model, torch = load_model(model_dir)
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Could not load a fine-tuned model: {exc}")
        st.info("Run the RoBERTa notebook, download its model archive, extract it, and select that folder above.")
    else:
        try:
            scores = probability_fn([email], tokenizer, model, torch, max_length)[0]
            phishing_probability = float(scores[1])
            label = "Likely phishing" if phishing_probability >= 0.5 else "Likely legitimate"
            st.subheader(label)
            st.metric("Phishing probability", f"{phishing_probability:.1%}")
            st.progress(phishing_probability)
            st.caption("This is a model estimate, not a security guarantee. Verify suspicious requests through a trusted channel.")

            from lime.lime_text import LimeTextExplainer

            explainer = LimeTextExplainer(class_names=["Legitimate", "Phishing"])
            with st.spinner("Generating LIME explanation…"):
                explanation = explainer.explain_instance(
                    email,
                    lambda samples: probability_fn(samples, tokenizer, model, torch, max_length),
                    labels=(0, 1),
                    num_features=num_features,
                    num_samples=1000,
                )
            st.subheader("Words influencing this prediction")
            st.caption("Green words support the selected class; red words count against it.")
            ranked_terms = explanation.as_list(label=1 if phishing_probability >= 0.5 else 0)
            st.code(
                "\n".join(f"{term[:42]:42s} {weight:+.4f}" for term, weight in ranked_terms),
                language=None,
            )
        except (ImportError, OSError, ValueError, RuntimeError) as exc:
            st.error(f"Analysis could not be completed: {exc}")

st.divider()
st.caption("For research and demonstration only. Do not rely on this prototype as the sole defense against phishing.")
