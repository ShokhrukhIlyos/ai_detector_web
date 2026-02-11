import os
import re
import math
import statistics
import tkinter as tk
from tkinter import filedialog, messagebox
from dataclasses import dataclass

from docx import Document
from pypdf import PdfReader

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SUPPORTED_EXT = [("Supported files", "*.docx *.pdf *.txt")]


# ----------------------------
# FILE READERS
# ----------------------------

def read_txt(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(path):
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(path):
    reader = PdfReader(path)
    text = ""
    for page in reader.pages:
        t = page.extract_text()
        if t:
            text += t + "\n"
    return text

def load_text(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        return read_pdf(path)
    raise ValueError("Unsupported file type")


# ----------------------------
# TEXT PROCESSING
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


# ----------------------------
# PERPLEXITY MODEL
# ----------------------------

class PerplexityScorer:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def perplexity(self, text):
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(self.device)
        outputs = self.model(input_ids, labels=input_ids)
        loss = outputs.loss
        return float(math.exp(loss))


# ----------------------------
# FEATURE CALCULATION
# ----------------------------

def compute_features(text, scorer):
    text = clean_text(text)
    sentences = split_sentences(text)
    words = tokenize_words(text)

    # Burstiness
    lengths = [len(tokenize_words(s)) for s in sentences if len(s.strip()) > 0]
    if len(lengths) > 1 and statistics.mean(lengths) > 0:
        burst = statistics.pstdev(lengths) / statistics.mean(lengths)
    else:
        burst = 0

    # Repetition (3-gram)
    tokens = tokenize_words(text)
    trigrams = list(zip(tokens, tokens[1:], tokens[2:])) if len(tokens) >= 3 else []
    counts = {}
    for tg in trigrams:
        counts[tg] = counts.get(tg, 0) + 1
    repeated = sum(1 for v in counts.values() if v > 1)
    repetition = repeated / max(len(counts), 1) if counts else 0

    # Unique ratio
    unique_ratio = len(set(words)) / len(words) if words else 0

    ppl = scorer.perplexity(text) if len(text) > 100 else 9999

    return Features(ppl, burst, repetition, unique_ratio)


def clamp(x, lo=0, hi=1):
    return max(lo, min(hi, x))


def estimate_ai_percent(f):
    # Perplexity normalization
    if f.perplexity <= 40:
        ppl_score = 1
    elif f.perplexity >= 120:
        ppl_score = 0
    else:
        ppl_score = 1 - ((f.perplexity - 40) / 80)

    burst_score = 1 - clamp((f.burstiness - 0.15) / 0.7)
    rep_score = clamp((f.repetition - 0.02) / 0.2)
    uniq_score = 1 - clamp((f.unique_ratio - 0.35) / 0.3)

    score = (
        0.55 * ppl_score +
        0.20 * burst_score +
        0.15 * rep_score +
        0.10 * uniq_score
    )

    return round(clamp(score) * 100, 2)


# ----------------------------
# GUI
# ----------------------------

def select_file():
    filepath = filedialog.askopenfilename(filetypes=SUPPORTED_EXT)
    if not filepath:
        return

    try:
        status_label.config(text="Loading model... please wait ⏳")
        root.update()

        scorer = PerplexityScorer()

        status_label.config(text="Reading file...")
        root.update()
        text = load_text(filepath)

        status_label.config(text="Analyzing...")
        root.update()
        features = compute_features(text, scorer)

        ai_percent = estimate_ai_percent(features)

        messagebox.showinfo(
            "AI Detection Result",
            f"Estimated AI Likelihood: {ai_percent}%\n\n"
            f"Perplexity: {features.perplexity:.2f}\n"
            f"Burstiness: {features.burstiness:.3f}\n"
            f"Repetition: {features.repetition:.3f}\n"
            f"Unique Ratio: {features.unique_ratio:.3f}\n\n"
            "⚠️ This is heuristic, not proof."
        )

        status_label.config(text="Done ✔")

    except Exception as e:
        messagebox.showerror("Error", str(e))


# Create GUI
root = tk.Tk()
root.title("AI Text Detector")
root.geometry("400x200")

title = tk.Label(root, text="AI Likelihood Detector", font=("Arial", 14))
title.pack(pady=15)

btn = tk.Button(root, text="Select File", command=select_file, height=2, width=20)
btn.pack()

status_label = tk.Label(root, text="")
status_label.pack(pady=15)

root.mainloop()
