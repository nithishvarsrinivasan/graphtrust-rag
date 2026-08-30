"""
config.py
---------
Extracted from your notebook's CONFIG section (cells 6-9), unchanged
logic - just centralized and no longer dependent on being run inside a
Colab cell.

The one behavior change: OPENROUTER_API_KEY now reads from a plain
environment variable instead of `google.colab.userdata`. That Colab-only
call still happens in your notebook's setup cell (see stage0_setup.ipynb
below) - it just sets the same env var, so this file works identically
whether it's imported in Colab or run locally.
"""

import os

# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# Dataset
DATASET_NAME = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
DATASET_SPLIT = "validation"

NUM_CORPUS_PASSAGES = 500
NUM_TEST_QUESTIONS = 20

CORPUS_JSONL_PATH = "./data/corpus.jsonl"
QUESTIONS_JSONL_PATH = "./data/questions.jsonl"

# ---------------------------------------------------------------------------
# Retriever (ColBERT)
COLBERT_CHECKPOINT = "colbert-ir/colbertv2.0"
COLBERT_INDEX_NAME = "graphtrust_base_index"
COLBERT_INDEX_ROOT = "./indexes"
TOP_K = 5

# ---------------------------------------------------------------------------
# Generator (Nemotron 3 Ultra via OpenRouter)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GENERATOR_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"

GENERATOR_MAX_TOKENS = 1536  # bumped from 512 per your earlier fix
GENERATOR_TEMPERATURE = 0.0

# ---------------------------------------------------------------------------
# Stage 1: Attack simulation
# A target question from your questions.jsonl to attack, and the false
# answer the adversarial passage will try to steer the model toward.
# TODO: pick one of your actual questions/answers here before running attack.py
ATTACK_TARGET_QUESTION_ID = None   # e.g. "5ae6b6065542991bbc976168"
ATTACK_TARGET_FALSE_ANSWER = None  # e.g. "Tom Hardy" (a wrong but plausible answer)

ATTACK_NUM_ROUNDS = 5              # how many refinement rounds to run
ATTACK_SIMILARITY_MODEL = "all-MiniLM-L6-v2"  # small, CPU-friendly, no GPU needed
