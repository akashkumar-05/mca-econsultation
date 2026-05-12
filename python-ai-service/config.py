"""
Configuration module for MCA eConsultation AI Service.

Centralizes all hyperparameters, paths, and constants for reproducibility.
All configuration is defined here to ensure experiments are reproducible
and parameters are easily tunable for ablation studies.
"""

import os
import torch

# ─────────────────────────────────────────────────────────────
# HARDWARE CONFIGURATION
# ─────────────────────────────────────────────────────────────
# Automatically detect GPU availability; fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if torch.backends.mps.is_available() else "cpu")

# ─────────────────────────────────────────────────────────────
# PATH CONFIGURATION
# ─────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
LOG_DIR = os.path.join(BASE_DIR, "logs")
EVAL_DIR = os.path.join(BASE_DIR, "evaluation")
MODEL_DIR = os.path.join(BASE_DIR, "policy_sentiment_final")

os.makedirs(STATIC_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# SENTIMENT MODEL CONFIGURATION
# ─────────────────────────────────────────────────────────────
FALLBACK_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_MAP_REVERSE = {v: k for k, v in LABEL_MAP.items()}

SENTIMENT_BATCH_SIZE = 32          # Batch size for inference
SENTIMENT_MAX_LENGTH = 256         # Max token length for RoBERTa
UNCERTAINTY_THRESHOLD = 0.6        # Below this → flagged as uncertain

# ─────────────────────────────────────────────────────────────
# SUMMARIZATION CONFIGURATION
# ─────────────────────────────────────────────────────────────
SUMMARIZER_MODEL_NAME = "facebook/bart-large-cnn"
SUMMARIZER_MAX_INPUT_TOKENS = 1024  # BART max input
SUMMARIZER_CHUNK_SIZE = 800         # Tokens per chunk for chunked summarization
SUMMARIZER_MAX_OUTPUT = 180         # Max summary tokens
SUMMARIZER_MIN_OUTPUT = 80          # Min summary tokens
SUMMARIZER_NUM_BEAMS = 4
SUMMARIZER_LENGTH_PENALTY = 2.0
SUMMARIZER_MAX_COMMENTS = 50       # Max comments to consider (was 20)
SUMMARIZER_MAX_CHARS = 5000        # Max chars total (was 3500)

# ─────────────────────────────────────────────────────────────
# API CONFIGURATION
# ─────────────────────────────────────────────────────────────
API_PORT = int(os.environ.get("PORT", 5001))
API_HOST = "0.0.0.0"
MAX_INPUT_TEXTS = 5000  # Max texts per request
