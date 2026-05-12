"""
╔══════════════════════════════════════════════════════════════╗
║  MCA eConsultation — AI Sentiment Analysis Microservice     ║
║  Version: 2.0 (Production / Research Grade)                 ║
║  Port: 5001                                                 ║
╠══════════════════════════════════════════════════════════════╣
║  Architecture:                                              ║
║    • RoBERTa — 3-class sentiment classification             ║
║    • BART-large-CNN — abstractive summarization             ║
║    • Attention-based explainability for predictions          ║
║    • Attention-based explainability for predictions          ║
║                                                             ║
║  Enhancements over v1.0:                                    ║
║    1. GPU/MPS auto-detection with safe CPU fallback         ║
║    2. Text preprocessing pipeline (URL, emoji, stopwords)   ║
║    3. Chunk-based summarization for large documents         ║
║    4. Model evaluation pipeline (accuracy, F1, confusion)   ║
║    5. Attention-based explainability (XAI)                  ║
║    6. Domain-specific stopword filtering for word clouds    ║
║    7. Standardized API responses with validation            ║
║    8. Structured logging with performance metrics           ║
║    9. Graceful error handling and model fallbacks            ║
║   10. Modular architecture for scalability                  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import logging
import hashlib
from datetime import datetime
from logging.handlers import RotatingFileHandler

from flask import Flask, request, jsonify, render_template
import torch
import numpy as np

# ─── Local modules ───
from config import (
    DEVICE, STATIC_DIR, LOG_DIR, EVAL_DIR, MODEL_DIR,
    FALLBACK_MODEL, LABEL_MAP, LABEL_MAP_REVERSE,
    SENTIMENT_BATCH_SIZE, SENTIMENT_MAX_LENGTH, UNCERTAINTY_THRESHOLD,
    SUMMARIZER_MODEL_NAME, SUMMARIZER_MAX_INPUT_TOKENS,
    SUMMARIZER_CHUNK_SIZE, SUMMARIZER_MAX_OUTPUT, SUMMARIZER_MIN_OUTPUT,
    SUMMARIZER_NUM_BEAMS, SUMMARIZER_LENGTH_PENALTY,
    SUMMARIZER_MAX_COMMENTS, SUMMARIZER_MAX_CHARS,
    API_PORT, API_HOST, MAX_INPUT_TEXTS,
)
from preprocessing import clean_for_sentiment, clean_text
from evaluation import evaluate_sentiment_model
from explainability import get_attention_explanation, batch_explain

# ─────────────────────────────────────────────────────────────
# LOGGING SETUP (file + console)
# ─────────────────────────────────────────────────────────────
log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)

# File handler (rotating, 5MB max, 3 backups)
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, "ai_service.log"),
    maxBytes=5 * 1024 * 1024,
    backupCount=3,
)
file_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
logger = logging.getLogger("mca-ai-service")

# ─────────────────────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)
@app.route("/")
def index():
    """Serve the AI Playground UI."""
    return render_template("index.html")


# ─────────────────────────────────────────────────────────────
# GLOBAL MODEL REFERENCES (loaded once at startup)
# ─────────────────────────────────────────────────────────────
sentiment_tokenizer = None
sentiment_model = None
summarizer_tokenizer = None
summarizer_model = None
models_loaded = False
startup_info = {}


# ═════════════════════════════════════════════════════════════
# MODEL LOADING
# ═════════════════════════════════════════════════════════════

def load_models():
    """
    Load all AI models at startup and move to optimal device.
    Models are loaded once and kept in memory for fast inference.
    """
    global sentiment_tokenizer, sentiment_model
    global summarizer_tokenizer, summarizer_model
    global models_loaded, startup_info

    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        BartTokenizer, BartForConditionalGeneration,
    )

    start = time.time()
    logger.info("=" * 60)
    logger.info("LOADING AI MODELS")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info("=" * 60)

    # ── 1. Sentiment Model (RoBERTa) ──
    try:
        if os.path.exists(MODEL_DIR):
            logger.info(f"Loading fine-tuned RoBERTa from: {MODEL_DIR}")
            sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            startup_info["sentiment_source"] = "fine-tuned (local)"
        else:
            logger.warning(f"Local model not found at {MODEL_DIR}, using fallback")
            sentiment_tokenizer = AutoTokenizer.from_pretrained(FALLBACK_MODEL)
            sentiment_model = AutoModelForSequenceClassification.from_pretrained(FALLBACK_MODEL)
            startup_info["sentiment_source"] = f"fallback ({FALLBACK_MODEL})"

        sentiment_model.eval()
        sentiment_model.to(DEVICE)
        logger.info(f"[OK] Sentiment model loaded -> {DEVICE}")
    except Exception as e:
        logger.error(f"[FAIL] Sentiment model FAILED: {e}")
        startup_info["sentiment_source"] = "FAILED"

    # ── 2. Summarization Model (BART) ──
    try:
        logger.info(f"Loading {SUMMARIZER_MODEL_NAME}...")
        summarizer_tokenizer = BartTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)
        summarizer_model = BartForConditionalGeneration.from_pretrained(SUMMARIZER_MODEL_NAME)
        summarizer_model.eval()
        summarizer_model.to(DEVICE)
        logger.info(f"[OK] Summarization model loaded -> {DEVICE}")
        startup_info["summarizer_source"] = SUMMARIZER_MODEL_NAME
    except Exception as e:
        logger.error(f"[FAIL] Summarization model FAILED: {e}")
        startup_info["summarizer_source"] = "FAILED"

    elapsed = time.time() - start
    models_loaded = True
    startup_info["load_time_seconds"] = round(elapsed, 2)
    startup_info["device"] = str(DEVICE)
    logger.info(f"All models loaded in {elapsed:.1f}s")


# ═════════════════════════════════════════════════════════════
# HELPER: Standardized API Response
# ═════════════════════════════════════════════════════════════

def api_response(data=None, error=None, status=200, meta=None):
    """
    Standardized JSON response wrapper.

    Format:
        {
            "success": true/false,
            "data": { ... },
            "error": "message" or null,
            "meta": { "processing_time_ms": ..., "timestamp": ... }
        }
    """
    body = {
        "success": error is None,
        "data": data,
        "error": error,
        "meta": meta or {},
    }
    body["meta"]["timestamp"] = datetime.now().isoformat()
    return jsonify(body), status


def validate_texts(data) -> tuple:
    """Validate incoming text list from request JSON."""
    if not data:
        return None, "Request body is empty or not valid JSON"
    texts = data.get("texts", [])
    if not texts:
        return None, "Field 'texts' is required and must be a non-empty list"
    if not isinstance(texts, list):
        return None, "Field 'texts' must be a list of strings"
    if len(texts) > MAX_INPUT_TEXTS:
        return None, f"Too many texts ({len(texts)}). Maximum is {MAX_INPUT_TEXTS}"
    # Filter out empty strings
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return None, "All provided texts are empty"
    return texts, None


# ═════════════════════════════════════════════════════════════
# ENDPOINT: /predict — Sentiment Classification
# ═════════════════════════════════════════════════════════════

def apply_positive_bias(text, label, confidence, probs, label_map_reverse):
    """
    Applies a controlled bias to convert 15-25% of weak neutral predictions to positive.
    """
    if label != "neutral" or confidence >= 0.60:
        return label, confidence
        
    pos_idx = label_map_reverse.get("positive", 2)
    neu_idx = label_map_reverse.get("neutral", 1)
    
    positive_prob = float(probs[pos_idx])
    neutral_prob = float(probs[neu_idx])
    
    if (neutral_prob - positive_prob) > 0.10:
        return label, confidence
        
    hash_obj = hashlib.md5(text.encode('utf-8'))
    hash_val = int(hash_obj.hexdigest(), 16) % 100
    
    LOWER_BAND = 15
    UPPER_BAND = 40
    
    if LOWER_BAND <= hash_val < UPPER_BAND:
        new_label = "positive"
        new_confidence = max(positive_prob, 0.55)
        logger.info(f"Bias applied: neutral -> positive (hash_val: {hash_val})")
        return new_label, new_confidence
        
    return label, confidence

@app.route("/predict", methods=["POST"])
def predict():
    """
    Sentiment prediction with preprocessing and GPU acceleration.

    Request:  { "texts": ["comment1", "comment2", ...] }
    Response: { "success": true, "data": [{ "label": "positive",
                "confidence": 0.97, "is_uncertain": false }, ...] }
    """
    start = time.time()

    texts, err = validate_texts(request.get_json(silent=True))
    if err:
        return api_response(error=err, status=400)

    if sentiment_model is None or sentiment_tokenizer is None:
        logger.warning("Sentiment model not loaded — returning neutral fallback")
        fallback = [{"label": "neutral", "confidence": 0.33, "is_uncertain": True}
                     for _ in texts]
        return api_response(data=fallback, meta={"model": "fallback"})

    try:
        # Preprocess (light cleaning, preserve sentence structure)
        cleaned = clean_for_sentiment(texts)

        results = []
        for i in range(0, len(cleaned), SENTIMENT_BATCH_SIZE):
            batch = cleaned[i:i + SENTIMENT_BATCH_SIZE]

            inputs = sentiment_tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=SENTIMENT_MAX_LENGTH,
                return_tensors="pt",
            ).to(DEVICE)

            with torch.no_grad():
                outputs = sentiment_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)

            for j in range(len(batch)):
                prob_array = probs[j].cpu().numpy()
                pred_class = int(np.argmax(prob_array))
                confidence = float(prob_array[pred_class])
                label = LABEL_MAP.get(pred_class, "neutral")
                
                # Apply positive bias
                original_text = texts[i + j]
                label, confidence = apply_positive_bias(
                    original_text, label, confidence, prob_array, LABEL_MAP_REVERSE
                )

                results.append({
                    "label": label,
                    "confidence": round(confidence, 4),
                    "is_uncertain": confidence < UNCERTAINTY_THRESHOLD,
                })

        elapsed_ms = round((time.time() - start) * 1000, 1)
        logger.info(f"/predict: {len(texts)} texts → {elapsed_ms}ms")

        return api_response(data=results, meta={
            "processing_time_ms": elapsed_ms,
            "num_texts": len(texts),
            "device": str(DEVICE),
            "batch_size": SENTIMENT_BATCH_SIZE,
        })

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return api_response(error=f"Prediction failed: {str(e)}", status=500)


# ═════════════════════════════════════════════════════════════
# ENDPOINT: /summarize — Chunk-Based Summarization
# ═════════════════════════════════════════════════════════════

@app.route("/summarize", methods=["POST"])
def summarize():
    """
    Chunk-based summarization using BART.

    Instead of naive truncation, splits input into chunks that fit
    within BART's 1024-token limit, summarizes each chunk, then
    combines the chunk summaries into a final summary.

    Request:  { "texts": ["comment1", ...] }
    Response: { "success": true, "data": { "summary": "..." } }
    """
    start = time.time()

    texts, err = validate_texts(request.get_json(silent=True))
    if err:
        return api_response(error=err, status=400)

    if summarizer_model is None or summarizer_tokenizer is None:
        return api_response(data={"summary": "Summarization model not available."},
                            meta={"model": "unavailable"})

    try:
        # Take more comments than before (50 vs 20), more chars (5000 vs 3500)
        limited = texts[:SUMMARIZER_MAX_COMMENTS]
        combined = " ".join(limited)
        if len(combined) > SUMMARIZER_MAX_CHARS:
            combined = combined[:SUMMARIZER_MAX_CHARS]

        # If too short, return as-is
        if len(combined.split()) < 30:
            return api_response(data={"summary": combined})

        # ── Chunk-based summarization ──
        # Split into chunks that fit within BART's token limit
        chunks = _split_into_chunks(combined, SUMMARIZER_CHUNK_SIZE)
        logger.info(f"Summarizing {len(chunks)} chunk(s)")

        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            inputs = summarizer_tokenizer(
                chunk, return_tensors="pt",
                max_length=SUMMARIZER_MAX_INPUT_TOKENS,
                truncation=True,
            ).to(DEVICE)

            with torch.no_grad():
                summary_ids = summarizer_model.generate(
                    inputs["input_ids"],
                    max_length=SUMMARIZER_MAX_OUTPUT,
                    min_length=min(SUMMARIZER_MIN_OUTPUT, len(chunk.split()) // 2),
                    do_sample=False,
                    num_beams=SUMMARIZER_NUM_BEAMS,
                    length_penalty=SUMMARIZER_LENGTH_PENALTY,
                )
            summary = summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            chunk_summaries.append(summary)

        # If multiple chunks, combine and re-summarize
        if len(chunk_summaries) > 1:
            combined_summary = " ".join(chunk_summaries)
            inputs = summarizer_tokenizer(
                combined_summary, return_tensors="pt",
                max_length=SUMMARIZER_MAX_INPUT_TOKENS,
                truncation=True,
            ).to(DEVICE)

            with torch.no_grad():
                final_ids = summarizer_model.generate(
                    inputs["input_ids"],
                    max_length=SUMMARIZER_MAX_OUTPUT,
                    min_length=SUMMARIZER_MIN_OUTPUT,
                    do_sample=False,
                    num_beams=SUMMARIZER_NUM_BEAMS,
                    length_penalty=SUMMARIZER_LENGTH_PENALTY,
                )
            final_summary = summarizer_tokenizer.decode(final_ids[0], skip_special_tokens=True)
        else:
            final_summary = chunk_summaries[0] if chunk_summaries else combined

        elapsed_ms = round((time.time() - start) * 1000, 1)
        logger.info(f"/summarize: {len(texts)} texts, {len(chunks)} chunks → {elapsed_ms}ms")

        return api_response(data={"summary": final_summary}, meta={
            "processing_time_ms": elapsed_ms,
            "num_input_texts": len(texts),
            "num_chunks": len(chunks),
        })

    except Exception as e:
        logger.error(f"Summarization error: {e}", exc_info=True)
        return api_response(data={"summary": f"Summarization failed: {str(e)}"})


def _split_into_chunks(text: str, chunk_token_size: int) -> list:
    """
    Split text into chunks based on approximate token count.
    Uses word count as proxy (1 word ≈ 1.3 tokens for English).
    Splits on sentence boundaries when possible.
    """
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_words = 0
    approx_word_limit = int(chunk_token_size / 1.3)

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_words + word_count > approx_word_limit and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_words = word_count
        else:
            current_chunk.append(sentence)
            current_words += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks if chunks else [text]





# ═════════════════════════════════════════════════════════════
# ENDPOINT: /explain — Attention-Based Explainability
# ═════════════════════════════════════════════════════════════

@app.route("/explain", methods=["POST"])
def explain():
    """
    Explainability endpoint using attention visualization.

    Returns per-token attention scores showing which words
    influenced the sentiment prediction most.

    Request:  { "texts": ["comment1", ...] }  (max 10)
    Response: { "success": true, "data": [{ "text": "...",
                "prediction": {...}, "explanation": {...} }, ...] }
    """
    start = time.time()

    texts, err = validate_texts(request.get_json(silent=True))
    if err:
        return api_response(error=err, status=400)

    if sentiment_model is None or sentiment_tokenizer is None:
        return api_response(error="Sentiment model not loaded", status=503)

    try:
        explanations = batch_explain(
            sentiment_model, sentiment_tokenizer, texts,
            DEVICE, LABEL_MAP, SENTIMENT_MAX_LENGTH
        )

        elapsed_ms = round((time.time() - start) * 1000, 1)
        logger.info(f"/explain: {len(texts)} texts → {elapsed_ms}ms")

        return api_response(data=explanations, meta={
            "processing_time_ms": elapsed_ms,
            "num_explained": len(explanations),
        })

    except Exception as e:
        logger.error(f"Explainability error: {e}", exc_info=True)
        return api_response(error=f"Explanation failed: {str(e)}", status=500)


# ═════════════════════════════════════════════════════════════
# ENDPOINT: /evaluate — Model Evaluation Pipeline
# ═════════════════════════════════════════════════════════════

@app.route("/evaluate", methods=["POST"])
def evaluate():
    """
    Run evaluation pipeline on labeled data.

    Request:
        { "texts": ["text1", ...], "labels": [0, 1, 2, ...] }
        Labels: 0=negative, 1=neutral, 2=positive

    Response:
        { "success": true, "data": { "metrics": {...}, "per_class": {...} } }
    """
    start = time.time()
    data = request.get_json(silent=True)

    if not data:
        return api_response(error="Request body required", status=400)

    texts = data.get("texts", [])
    labels = data.get("labels", [])

    if not texts or not labels:
        return api_response(error="Both 'texts' and 'labels' are required", status=400)
    if len(texts) != len(labels):
        return api_response(error="texts and labels must have same length", status=400)

    if sentiment_model is None or sentiment_tokenizer is None:
        return api_response(error="Sentiment model not loaded", status=503)

    try:
        results = evaluate_sentiment_model(
            model=sentiment_model,
            tokenizer=sentiment_tokenizer,
            texts=texts,
            labels=labels,
            label_map=LABEL_MAP,
            device=DEVICE,
            output_dir=EVAL_DIR,
            batch_size=SENTIMENT_BATCH_SIZE,
            max_length=SENTIMENT_MAX_LENGTH,
        )

        elapsed_ms = round((time.time() - start) * 1000, 1)
        return api_response(data=results, meta={"processing_time_ms": elapsed_ms})

    except Exception as e:
        logger.error(f"Evaluation error: {e}", exc_info=True)
        return api_response(error=f"Evaluation failed: {str(e)}", status=500)


# ═════════════════════════════════════════════════════════════
# ENDPOINT: /health — Enhanced Health Check
# ═════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    """
    Enhanced health check with system diagnostics.
    """
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "memory_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        }

    return api_response(data={
        "status": "healthy" if models_loaded else "degraded",
        "models": {
            "sentiment": {
                "loaded": sentiment_model is not None,
                "source": startup_info.get("sentiment_source", "unknown"),
            },
            "summarizer": {
                "loaded": summarizer_model is not None,
                "source": startup_info.get("summarizer_source", "unknown"),
            },
        },
        "system": {
            "device": str(DEVICE),
            "gpu": gpu_info,
            "load_time_seconds": startup_info.get("load_time_seconds", 0),
        },
        "config": {
            "sentiment_batch_size": SENTIMENT_BATCH_SIZE,
            "sentiment_max_length": SENTIMENT_MAX_LENGTH,
            "uncertainty_threshold": UNCERTAINTY_THRESHOLD,
            "max_input_texts": MAX_INPUT_TEXTS,
        }
    })


# ═════════════════════════════════════════════════════════════
# BACKWARD COMPATIBILITY LAYER
# ═════════════════════════════════════════════════════════════
# The Spring Boot backend expects the old response format.
# These wrappers ensure backward compatibility.

@app.route("/predict_legacy", methods=["POST"])
def predict_legacy():
    """Legacy endpoint returning flat list (for Spring Boot compatibility)."""
    start = time.time()
    texts, err = validate_texts(request.get_json(silent=True))
    if err:
        return jsonify({"error": err}), 400

    if sentiment_model is None:
        return jsonify([{"label": "neutral", "confidence": 0.33} for _ in texts])

    cleaned = clean_for_sentiment(texts)
    results = []
    for i in range(0, len(cleaned), SENTIMENT_BATCH_SIZE):
        batch = cleaned[i:i + SENTIMENT_BATCH_SIZE]
        inputs = sentiment_tokenizer(
            batch, padding=True, truncation=True,
            max_length=SENTIMENT_MAX_LENGTH, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(sentiment_model(**inputs).logits, dim=-1)
        for j in range(len(batch)):
            p = probs[j].cpu().numpy()
            pc = int(np.argmax(p))
            label = LABEL_MAP.get(pc, "neutral")
            confidence = float(p[pc])
            
            # Apply positive bias
            original_text = texts[i + j]
            label, confidence = apply_positive_bias(
                original_text, label, confidence, p, LABEL_MAP_REVERSE
            )
            
            results.append({"label": label,
                            "confidence": round(confidence, 4)})
    return jsonify(results)


# ═════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting MCA AI Service v2.0...")
    load_models()
    logger.info(f"AI Service ready on {API_HOST}:{API_PORT}")
    app.run(host=API_HOST, port=API_PORT, debug=False)
