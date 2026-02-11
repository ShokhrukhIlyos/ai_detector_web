import streamlit as st
import os
import re
import math
import statistics
from dataclasses import dataclass

from docx import Document
from pypdf import PdfReader

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Page config
# ----------------------------
st.set_page_config(page_title="AI Detector", layout="centered")
st.title("🧠 AI Likelihood Detector (Heuristic)")
st.caption("Upload PDF/DOCX/TXT and get estimated AI percentage.")


# ----------------------------
# File readers
# ----------------------------
def read_docx(file):
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def load_text(uploaded_file):
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext == ".txt":
        return uploaded_file.read().decode("utf-8", errors="ignore")
    if ext == ".docx":
        return read_docx(uploaded_file)
    if ext == ".pdf":
        return read_pdf(uploaded_file)
    return ""


# ----------------------------
# Text processing
# ----------------------------
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def split_sentences(text):
    return re.split(r"(?<=[.!?])\s+", text)

def tokenize_words(text):
    return re.findall(r"[A-Za-z']+", text.lower())


@dataclass
class Features:
    perplexity: float
    burstiness: float
    repetition: float
    unique_ratio: float


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model.eval()
    return tokenizer, model


def compute_perplexity(text, tokenizer, model):
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
    input_ids = enc["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    return float(math.exp(outputs.loss))


def compute_features(text, tokenizer, model):
    text = clean_text(text)
    sentences = split_sentences(text)
    words = tokenize_words(text)

    lengths = [len(tokenize_words(s)) for s in sentences if s.strip()]
    if len(lengths) > 1 and statistics.mean(lengths) > 0:
        burst = statistics.pstdev(lengths) / statistics.mean(lengths)
    else:
        burst = 0

    unique_ratio = len(set(words)) / len(words) if words else 0

    ppl = compute_perplexity(text, tokenizer, model) if len(text) > 150 else 9999

    return Features(ppl, burst, 0, unique_ratio)


def estimate_ai_percent(f):
    if f.perplexity <= 40:
        ppl_score = 1
    elif f.perplexity >= 120:
        ppl_score = 0
    else:
        ppl_score = 1 - ((f.perplexity - 40) / 80)

    burst_score = 1 - min(max((f.burstiness - 0.15) / 0.7, 0), 1)
    uniq_score = 1 - min(max((f.unique_ratio - 0.35) / 0.3, 0), 1)

    score = 0.6 * ppl_score + 0.2 * burst_score + 0.2 * uniq_score
    return round(min(max(score, 0), 1) * 100, 2)


# ----------------------------
# UI
# ----------------------------
uploaded_file = st.file_uploader("Upload file", type=["pdf", "docx", "txt"])

if uploaded_file:
    with st.spinner("Loading model..."):
        tokenizer, model = load_model()

    text = load_text(uploaded_file)

    with st.spinner("Analyzing..."):
        features = compute_features(text, tokenizer, model)
        ai_percent = estimate_ai_percent(features)

    st.success("Analysis complete!")

    st.metric("Estimated AI Likelihood", f"{ai_percent}%")
    st.write("Perplexity:", round(features.perplexity, 2))
    st.write("Burstiness:", round(features.burstiness, 3))
    st.write("Unique Ratio:", round(features.unique_ratio, 3))

    st.warning("⚠️ This is heuristic, not proof.")
