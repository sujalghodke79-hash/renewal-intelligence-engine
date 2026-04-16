"""
Configuration module for Renewal Intelligence Engine.
Centralizes all settings, API keys, and constants.
"""

import os
from pathlib import Path
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Data file paths ---
ACCOUNTS_PATH = DATA_DIR / "accounts.csv"
USAGE_METRICS_PATH = DATA_DIR / "usage_metrics.csv"
SUPPORT_TICKETS_PATH = DATA_DIR / "support_tickets.csv"
CSM_NOTES_PATH = DATA_DIR / "csm_notes.txt"
NPS_RESPONSES_PATH = DATA_DIR / "nps_responses.csv"
CHANGELOG_PATH = DATA_DIR / "changelog.md"

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# --- LLM Settings ---
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.1-8b-instant"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "nvidia/nemotron-3-super-120b-a12b:free"

# --- Business Logic Constants ---
REFERENCE_DATE_STR = os.getenv("REFERENCE_DATE", "2026-04-15")
REFERENCE_DATE = date.fromisoformat(REFERENCE_DATE_STR)
RENEWAL_WINDOW_DAYS = 90
RENEWAL_CUTOFF_DATE = REFERENCE_DATE + timedelta(days=RENEWAL_WINDOW_DAYS)

# Risk tier thresholds
RISK_TIER_HIGH = 70
RISK_TIER_MEDIUM = 40

# Usage trend settings
RECENT_MONTHS = 3
OLDER_MONTHS = 3

# SDK Deprecation dates from changelog
SDK_V3_SUNSET_DATE = date(2026, 4, 30)
SDK_V3_SECURITY_PATCH_END = date(2026, 4, 30)
LEGACY_EDITOR_REMOVAL = "v4.4.0 (May 2026)"
REST_API_V2_SUNSET = date(2026, 4, 30)
LEGACY_WORKFLOW_EDIT_DEADLINE = date(2026, 2, 28)