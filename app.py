import os
import re
import math
import statistics
from dataclasses import dataclass

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from docx import Document
from pypdf import PdfReader

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Flask config
# ----------------------------
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "uploads")
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024  # 15 MB upload limit

ALLOWED_EXT = {".txt", ".docx", ".pdf"}


# ----------------------------
# File readers
# ----------------------------
def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_docx(path: str) -> str:
    doc = Document(path)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        parts.append(t)
    return "\n".join(parts)

def load_text(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".txt":
        return read_txt(path)
    if ext == ".docx":
        return read_docx(path)
    if ext == ".pdf":
        return read_pdf(path)
    raise ValueError("Unsupported file type")


# ----------------------------
# Text processing + features
# ----------------------------
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def split_sentences(text: str):
    sents = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sents if s.strip()]

def tokenize_words(text: str):
    return re.findall(r"[A-Za-z']+", text.lower())

def tokenize_tokens(text: str):
    # for trigram repetition
    return re.findall(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]", text.lower())


@dataclass
class Features:
    perplexity: float
    burstiness: float
    repetition: float
    unique_ratio: float


class PerplexityScorer:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        # quick perplexity (truncates long texts)
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = enc["input_ids"].to(self.device)
        out = self.model(input_ids, labels=input_ids)
        return float(math.exp(out.loss))


def compute_features(text: str, scorer: PerplexityScorer) -> Features:
    text = clean_text(text)
    sents = split_sentences(text)

    # Burstiness
    sent_lengths = [len(tokenize_words(s)) for s in sents]
    if len(sent_lengths) >= 2 and statistics.mean(sent_lengths) > 0:
        burst = statistics.pstdev(sent_lengths) / statistics.mean(sent_lengths)
    else:
        burst = 0.0

    # Repetition (3-gram)
    tokens = tokenize_tokens(text)
    trigrams = list(zip(tokens, tokens[1:], tokens[2:])) if len(tokens) >= 3 else []
    if trigrams:
        counts = {}
        for tg in trigrams:
            counts[tg] = counts.get(tg, 0) + 1
        repeated = sum(1 for v in counts.values() if v >= 2)
        repetition = repeated / max(len(counts), 1)
    else:
        repetition = 0.0

    # Unique ratio (word diversity)
    words = tokenize_words(text)
    unique_ratio = (len(set(words)) / len(words)) if words else 0.0

    ppl = scorer.perplexity(text) if len(text) >= 150 else 9999.0

    return Features(perplexity=ppl, burstiness=float(burst), repetition=float(repetition), unique_ratio=float(unique_ratio))


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def estimate_ai_percent(f: Features) -> float:
    # Perplexity score: lower ppl => more AI-like
    if f.perplexity <= 40:
        ppl_score = 1.0
    elif f.perplexity >= 120:
        ppl_score = 0.0
    else:
        ppl_score = 1.0 - ((f.perplexity - 40) / (120 - 40))

    # Burstiness: lower => more AI-like
    burst_score = 1.0 - clamp((f.burstiness - 0.15) / (0.85 - 0.15))

    # Repetition: higher => more AI-like
    rep_score = clamp((f.repetition - 0.02) / (0.25 - 0.02))

    # Unique ratio: lower => more AI-like
    uniq_score = 1.0 - clamp((f.unique_ratio - 0.35) / (0.65 - 0.35))

    ai_score = (
        0.55 * ppl_score +
        0.20 * burst_score +
        0.15 * rep_score +
        0.10 * uniq_score
    )
    return round(clamp(ai_score) * 100, 2)


# Load model once at startup (faster for users)
SCORER = PerplexityScorer("gpt2")


# ----------------------------
# Routes
# ----------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", result=None, error=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return render_template("index.html", result=None, error="No file uploaded.")

    f = request.files["file"]
    if not f.filename:
        return render_template("index.html", result=None, error="Please choose a file.")

    filename = secure_filename(f.filename)
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return render_template("index.html", result=None, error="Unsupported file type. Use PDF/DOCX/TXT.")

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    saved_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    f.save(saved_path)

    try:
        text = load_text(saved_path)
        text = clean_text(text)

        features = compute_features(text, SCORER)
        ai_percent = estimate_ai_percent(features)

        result = {
            "filename": filename,
            "ai_percent": ai_percent,
            "perplexity": round(features.perplexity, 2),
            "burstiness": round(features.burstiness, 3),
            "repetition": round(features.repetition, 3),
            "unique_ratio": round(features.unique_ratio, 3),
            "chars": len(text),
        }

        return render_template("index.html", result=result, error=None)

    except Exception as e:
        return render_template("index.html", result=None, error=str(e))

    finally:
        # optional cleanup
        try:
            os.remove(saved_path)
        except Exception:
            pass


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
