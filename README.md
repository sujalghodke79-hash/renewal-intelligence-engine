# `README.md` — Complete Documentation

```markdown
# 🔮 Renewal Intelligence Engine

> An AI-powered renewal risk assessment system for B2B SaaS accounts.
> Built for Contentstack's BizOps team to identify at-risk renewals,
> explain why accounts are at risk, and surface non-obvious churn signals
> — before the account team has to ask.

---

## 📋 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Tech Stack](#3-tech-stack)
4. [Project Structure](#4-project-structure)
5. [Quick Start](#5-quick-start)
6. [Step-by-Step Setup](#6-step-by-step-setup)
7. [Running the Pipeline](#7-running-the-pipeline)
8. [CLI Reference](#8-cli-reference)
9. [Streamlit Dashboard](#9-streamlit-dashboard)
10. [How It Works — Module by Module](#10-how-it-works--module-by-module)
11. [Data Sources & Inconsistencies Handled](#11-data-sources--inconsistencies-handled)
12. [Risk Scoring Methodology](#12-risk-scoring-methodology)
13. [LangChain & LangGraph Usage](#13-langchain--langgraph-usage)
14. [Non-Obvious Insights](#14-non-obvious-insights)
15. [Output Files](#15-output-files)
16. [Tradeoffs & Design Decisions](#16-tradeoffs--design-decisions)
17. [What I'd Do With More Time](#17-what-id-do-with-more-time)
18. [What I'd Change for Production](#18-what-id-change-for-production)
19. [Troubleshooting](#19-troubleshooting)
20. [API Keys & Cost Estimates](#20-api-keys--cost-estimates)

---

## 1. Project Overview

### The Problem

Every quarter, Contentstack's BizOps team scrambles to figure out which
accounts are likely to churn or downgrade — relying on gut feel, scattered
Salesforce notes, and last-minute Slack threads. By the time a risk is
identified, it's often too late to act.

### The Solution

The Renewal Intelligence Engine ingests 5 data sources, reconciles
inconsistencies using LLMs, computes quantitative risk signals, and
produces:

- ✅ A risk-scored list of all accounts renewing in the next 90 days
- ✅ Plain-English explanations of WHY each account is at risk
- ✅ Specific, actionable next steps for each account team
- ✅ Non-obvious insights that rule-based systems would miss
- ✅ Executive summary for leadership
- ✅ Per-CSM briefings for weekly planning

### Key Design Principle

> **Conservative by default**: when data sources conflict, we trust the
> more negative signal. It is better to over-flag a healthy account than
> to miss a churning one.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    DATA SOURCES                                 │
│  accounts.csv  usage_metrics.csv  support_tickets.csv           │
│  csm_notes.txt  nps_responses.csv  changelog.md                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              PART 1: DATA INGESTION                             │
│  • Load all files with correct dtypes                           │
│  • Initial type validation                                      │
│  • Cross-reference checks                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│         PART 2: DATA RECONCILIATION (LangChain)                 │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ CSM Notes       │  │ NPS Processing   │  │ SDK Risk      │  │
│  │ Parsing (LLM)   │  │ Translation(LLM) │  │ Assessment    │  │
│  │ Fuzzy Matching  │  │ Contradiction     │  │ vs Changelog  │  │
│  │ Name → ID       │  │ Detection (LLM)  │  │               │  │
│  └────────┬────────┘  └────────┬─────────┘  └───────┬───────┘  │
│           └───────────────────┼────────────────────┘           │
│                               ▼                                 │
│                  Cross-Source Conflict Detection                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              PART 3: FEATURE ENGINEERING                        │
│  • Usage trend features (7 metrics × 8 computations)           │
│  • Support ticket features (volume, severity, status)          │
│  • NPS features (score, category, contradiction flags)         │
│  • CSM features (sentiment, competitors, signals)              │
│  • SDK/Platform features                                        │
│  • Contract/Firmographic features                               │
│  • Conflict features                                            │
│  OUTPUT: Master feature matrix (~108 features per account)     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           PART 4: LLM DEEP ANALYSIS (LangChain)                 │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ Changelog       │  │ Silent Churn     │  │ Portfolio     │  │
│  │ Impact Analysis │  │ Pattern          │  │ Insights      │  │
│  │ per account     │  │ Detection (LLM)  │  │ (LLM)        │  │
│  └─────────────────┘  └──────────────────┘  └───────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│           PART 5: RISK SCORING (LangGraph)                      │
│                                                                 │
│  quantitative_score → qualitative_assessment (LLM)             │
│  → conflict_resolution → final_risk → confidence               │
│                                                                 │
│  OUTPUT: Risk score (0-100) + Tier (High/Medium/Low)           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│         PART 6: EXPLANATION GENERATION (LangChain)             │
│  • Per-account risk narratives (LLM)                           │
│  • Executive summary (LLM)                                     │
│  • CSM briefings (LLM)                                         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              PART 7: INTERFACES                                 │
│         CLI (Rich terminal)  +  Streamlit Dashboard            │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph State Machine (Part 5)

```
AccountRiskState
      │
      ▼
[quantitative_score] ──── Weighted health scores + urgency/ARR multipliers
      │
      ▼
[qualitative_assessment] ── LLM: nuances, competitors, org changes → ±15 adjustment
      │
      ▼
[conflict_resolution] ──── Trust hierarchy when sources disagree → 0-20 adjustment
      │
      ▼
[final_risk] ────────────── Combine + override rules + recommended actions
      │
      ▼
[confidence] ────────────── Data completeness + signal agreement → high/medium/low
      │
      ▼
    [END]
```

---

## 3. Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Language | Python 3.11+ | Primary language |
| LLM Orchestration | LangChain 0.3.x | Prompt templates, chains, output parsers |
| Agentic Pipeline | LangGraph 0.2.x | Stateful multi-step risk assessment graph |
| LLM Provider (fast) | Groq (llama-3.1-8b-instant) | CSM parsing, NPS translation, scoring |
| LLM Provider (capable) | OpenRouter (llama-3.1-70b) | Portfolio insights, complex analysis |
| Data Processing | Pandas, NumPy | Feature engineering, data manipulation |
| Fuzzy Matching | FuzzyWuzzy + python-Levenshtein | Account name reconciliation |
| CLI | Rich | Beautiful terminal output |
| Dashboard | Streamlit + Plotly | Interactive web UI |
| Output Parsing | Pydantic v2 | Structured LLM output validation |

---

## 4. Project Structure

```
renewal-intelligence-engine/
│
├── data/                          # Input data files
│   ├── accounts.csv               # 120 accounts with firmographic/contract data
│   ├── usage_metrics.csv          # 6 months of product usage per account
│   ├── support_tickets.csv        # Support ticket history
│   ├── csm_notes.txt              # Unstructured CSM call notes (messy)
│   ├── nps_responses.csv          # NPS survey responses
│   └── changelog.md               # Product changelog (2 quarters)
│
├── src/                           # Source modules
│   ├── __init__.py
│   ├── config.py                  # All settings, paths, constants
│   ├── data_ingestion.py          # Load and validate all data files
│   ├── utils.py                   # LangChain LLM client factory
│   ├── data_reconciliation.py     # Reconcile + clean all data sources
│   ├── feature_engineering.py     # Compute 108 risk features per account
│   ├── llm_pipeline.py            # Deep LLM analysis (changelog, silent churn)
│   ├── risk_scoring.py            # LangGraph risk assessment pipeline
│   └── explanation_generator.py   # Generate narratives and summaries
│
├── outputs/                       # Generated outputs (auto-created)
│   ├── risk_scored_accounts.csv   # Main output: all accounts with scores
│   ├── all_account_features.csv   # Full feature matrix (120 accounts)
│   ├── renewal_account_features.csv # Features for renewal window only
│   ├── executive_summary.md       # Leadership briefing
│   ├── account_narratives.md      # Per-account risk narratives
│   ├── csm_briefings.md           # CSM-specific briefings
│   ├── changelog_impacts.csv      # Product change impacts per account
│   ├── silent_churn_analysis.csv  # Silent churn patterns
│   └── portfolio_insights.json    # Non-obvious portfolio insights
│
├── app.py                         # Streamlit dashboard
├── cli.py                         # CLI tool
│
├── test_part1.py                  # Test: data ingestion
├── test_part2.py                  # Test: data reconciliation
├── test_part3.py                  # Test: feature engineering
├── test_part4.py                  # Test: LLM pipeline
├── test_part5.py                  # Test: risk scoring
├── test_final.py                  # End-to-end test
│
├── requirements.txt               # Python dependencies
├── .env                           # API keys (not committed)
└── README.md                      # This file
```

---

## 5. Quick Start

If you want to get running as fast as possible:

```bash
# 1. Clone and enter the project
git clone <your-repo-url>
cd renewal-intelligence-engine

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
# OR
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure API keys
cp .env.example .env
# Edit .env and add your GROQ_API_KEY

# 5. Add your data files to data/
# (accounts.csv, usage_metrics.csv, etc.)

# 6. Run the pipeline
python cli.py run

# 7. View results
python cli.py summary
streamlit run app.py
```

---

## 6. Step-by-Step Setup

### Step 6.1 — Prerequisites

```bash
# Check Python version (need 3.11+)
python --version

# If needed, install Python 3.11 from https://python.org
```

### Step 6.2 — Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows (cmd)
venv\Scripts\Activate.ps1       # Windows (PowerShell)

# You should see (venv) in your terminal prompt
```

### Step 6.3 — Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
```
pandas==2.2.3
numpy==1.26.4
scikit-learn==1.5.2
streamlit==1.40.0
python-dotenv==1.0.1
httpx==0.27.2
openai==1.55.0
tiktoken==0.8.0
rich==13.9.4
tabulate==0.9.0
fuzzywuzzy==0.18.0
python-Levenshtein==0.26.1
plotly==5.24.1
langchain==0.3.14
langchain-core==0.3.28
langchain-groq==0.2.4
langchain-openai==0.3.0
langgraph==0.2.60
pydantic==2.10.4
```

**Expected output:**
```
Successfully installed langchain-0.3.14 langgraph-0.2.60 ...
```

### Step 6.4 — Configure Environment Variables

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env      # Mac/Linux
# OR manually create the file on Windows
```

Add your API keys:

```env
# .env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
REFERENCE_DATE=2026-04-15
```

**Getting API keys:**

**Groq (required, free):**
1. Go to https://console.groq.com
2. Sign up for free
3. Navigate to API Keys → Create API Key
4. Copy the key (starts with `gsk_`)

**OpenRouter (optional, free tier available):**
1. Go to https://openrouter.ai
2. Sign up
3. Navigate to Keys → Create Key
4. Copy the key (starts with `sk-or-v1-`)

> **Note:** The pipeline works with just `GROQ_API_KEY`. OpenRouter
> is only used if you explicitly pass `--provider openrouter`.

### Step 6.5 — Place Data Files

Put all 6 data files in the `data/` directory:

```
data/
├── accounts.csv
├── usage_metrics.csv
├── support_tickets.csv
├── csm_notes.txt
├── nps_responses.csv
└── changelog.md
```

### Step 6.6 — Verify Setup

Run a quick check to make sure everything is configured:

```bash
python test_part1.py
```

**Expected output:**
```
[Ingestion] Loaded 120 accounts
[Ingestion] Loaded XXXX usage metric rows
[Ingestion] Loaded XXX support tickets
[Ingestion] Loaded CSM notes: XXXX characters
[Ingestion] Loaded NPS responses: XXX responses

--- Cross-reference Check ---
Accounts: 120
Accounts with usage data: 120
Accounts with NPS: 98
...

✅ All Part 1 tests passed!
Accounts renewing between 2026-04-15 and 2026-07-14: XX
```

If you see this, you're ready to run the full pipeline.

---

## 7. Running the Pipeline

### Option A: CLI (Recommended for first run)

```bash
python cli.py run
```

This runs all 6 pipeline steps sequentially with progress output.

**What you'll see:**
```
╭──────────────────────────────────────────╮
│ 🚀 Starting Full Pipeline                │
│ Reference Date: 2026-04-15               │
│ Renewal Window: 2026-04-15 → 2026-07-14  │
│ LLM Provider: groq                       │
╰──────────────────────────────────────────╯

Step 1/6: Loading data...
✅ Step 1: Data loaded

Step 2/6: Reconciling data (LLM calls)...
  Parsing note 1/28...
  Parsing note 2/28...
  ...
  Fuzzy matched: 'BritePath Solutions' → 'BrightPath Solutions' (score=95)
  Fuzzy matched: 'Pinacle Media' → 'Pinnacle Media Group' (score=87)
  ⚠️  CONTRADICTION — Account 1019: Score=3, Comment='Great developer...'
✅ Step 2: Data reconciled

Step 3/6: Computing features...
✅ Step 3: Features computed

Step 4/6: LLM deep analysis (this takes a few minutes)...
  [1001] BrightPath Solutions: 3 silent churn patterns detected (risk=high)
  [1003] Meridian Health: 2 silent churn patterns detected (risk=high)
  ...
✅ Step 4: LLM analysis complete

Step 5/6: Scoring accounts with LangGraph...
  [1/XX] Processing Acme Corp (1000)...
    🟡 Risk: Medium (52.3/100) | Confidence: medium | ARR: $17,000
  ...
✅ Step 5: Risk scoring complete

Step 6/6: Generating explanations...
✅ Step 6: Explanations generated

Risk Tier Distribution:
  🔴 High:   X accounts | ARR: $X,XXX,XXX
  🟡 Medium: X accounts | ARR: $X,XXX,XXX
  🟢 Low:    X accounts | ARR: $X,XXX,XXX
```

**Estimated time:** 5-10 minutes (mostly LLM API calls)

### Option B: Streamlit (Run pipeline from UI)

```bash
streamlit run app.py
```

1. Open http://localhost:8501 in your browser
2. Navigate to **🚀 Run Pipeline** in the sidebar
3. Select your LLM provider
4. Click **Run Full Pipeline**

### Option C: Test Step-by-Step

Run each part independently to validate each module:

```bash
# Test each step individually
python test_part1.py   # Data ingestion (no LLM, fast)
python test_part2.py --quick   # Reconciliation without LLM calls
python test_part2.py   # Full reconciliation with LLM calls
python test_part3.py   # Feature engineering
python test_part4.py   # LLM deep analysis
python test_part5.py   # Risk scoring

# Or run the complete end-to-end test
python test_final.py
```

---

## 8. CLI Reference

### `python cli.py run`

Run the complete pipeline from scratch.

```bash
python cli.py run
python cli.py run --provider groq         # Use Groq (default)
python cli.py run --provider openrouter   # Use OpenRouter
```

### `python cli.py summary`

Display the executive summary and high-risk account table.

```bash
python cli.py summary
```

**Output example:**
```
┌─────────────────────────────────────────────┐
│          📊 Renewal Risk Summary            │
├──────────┬──────────┬────────────┬──────────┤
│ Tier     │ Accounts │ ARR        │ % of ARR │
├──────────┼──────────┼────────────┼──────────┤
│ 🔴 High  │ 8        │ $4,521,000 │ 43.2%    │
│ 🟡 Medium│ 12       │ $3,892,000 │ 37.2%    │
│ 🟢 Low   │ 15       │ $2,056,000 │ 19.6%    │
├──────────┼──────────┼────────────┼──────────┤
│ Total    │ 35       │ $10,469,000│ 100%     │
└──────────┴──────────┴────────────┴──────────┘
```

### `python cli.py list`

List all risk-scored accounts sorted by risk score (highest first).

```bash
python cli.py list
```

### `python cli.py account <ID>`

Show detailed risk profile for a specific account.

```bash
python cli.py account 1007    # Zenith Publishing
python cli.py account 1003    # Meridian Health
python cli.py account 1017    # Pacific Rim Trading
```

**Output includes:**
- Health score visualization (bar charts in terminal)
- Risk score waterfall breakdown
- Risk factors by severity
- Recommended actions
- Full risk narrative

**Example:**
```bash
python cli.py account 1007
```
```
╭──────────────────────────────────────────────────────────────╮
│ 🔴 Account Detail — Zenith Publishing (ID: 1007)             │
│                                                              │
│ ARR: $1,625,000 | Plan: Enterprise | Renewal: 20 days        │
│                                                              │
│ Risk Score: 87.4/100 → Tier: 🔴 High                        │
│ Confidence: high                                             │
│                                                              │
│ Health Scores:                                               │
│  Usage:   ████████░░░░░░░░░░░░ 42                           │
│  Support: ███████████░░░░░░░░░ 55                           │
│  NPS:     ████░░░░░░░░░░░░░░░░ 20                           │
│  CSM:     ██░░░░░░░░░░░░░░░░░░ 10                           │
│  SDK:     ████████████████████ 100                          │
│                                                              │
│ Competitors: Kontent.ai                                      │
│ Relationship: critical                                       │
│ Exec Attention: Yes ⚠️                                       │
╰──────────────────────────────────────────────────────────────╯
```

### `python cli.py csm "<Name>"`

Show a CSM-specific renewal briefing.

```bash
python cli.py csm "Sarah Chen"
python cli.py csm "James Okafor"
python cli.py csm "Priya Sharma"
```

**Output:** Markdown-formatted weekly briefing with account priorities,
escalation needs, and specific action items for that CSM.

---

## 9. Streamlit Dashboard

Launch the dashboard:

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Dashboard Pages

#### 📊 Overview
The main landing page showing:
- **KPI cards**: Total accounts, ARR, risk tier counts
- **ARR by Risk Tier** donut chart
- **Risk Score Distribution** histogram with tier thresholds
- **ARR vs Risk Score** scatter plot (interactive, hover for details)
- **Average Risk Score by CSM** bar chart (identify overloaded CSMs)
- **Renewal Timeline** scatter (when renewals land vs risk level)

#### 🔴 High Risk
Detailed expandable cards for each high-risk account:
- Health score metrics + progress bars
- Risk factors with severity labels
- Recommended actions
- Competitor flags

#### 🔍 Account Lookup
Search any renewal account by name/ID:
- **Waterfall chart**: How the final score is composed
  (Quantitative + Qualitative Adjustment + Conflict Adjustment)
- **Radar chart**: Health across all 5 dimensions
- **Tabbed detail view**: Risk Factors | Actions | Narrative | Raw Data

#### 📋 Executive Summary
- AI-generated portfolio briefing (Markdown rendered)
- Non-obvious portfolio insights (expandable cards with ARR at risk)
- Top priority action highlighted

#### 👤 CSM View
- Select any CSM from dropdown
- See their renewal account workload
- Read their weekly briefing
- Interactive account table with sorting

#### 🚀 Run Pipeline
- Run the full pipeline from the browser
- Select LLM provider
- Progress bar with status updates

### Sidebar Filters

All pages (except Account Lookup and Executive Summary) respect filters:
- **Risk Tier**: Filter by High/Medium/Low
- **CSM**: Filter to specific CSMs
- **Plan Tier**: Filter by Starter/Growth/Scale/Enterprise

---

## 10. How It Works — Module by Module

### `src/config.py` — Configuration

Centralizes all settings:
- File paths
- API keys (loaded from `.env`)
- Business logic constants (reference date, renewal window)
- Risk tier thresholds (High ≥ 70, Medium ≥ 40)
- SDK deprecation dates from changelog

**Key constant:**
```python
REFERENCE_DATE = date(2026, 4, 15)   # "Today" in the dataset timeline
RENEWAL_WINDOW_DAYS = 90             # Look 90 days ahead
```

### `src/data_ingestion.py` — Data Loading

Loads all 6 files with correct dtypes and initial validation:
- Converts `contract_end_date` to datetime
- Normalizes `month` column in usage metrics to proper dates
- Identifies non-English NPS comments
- Reports cross-reference gaps between files

### `src/data_reconciliation.py` — Data Cleaning (LangChain)

**LangChain usage here:**

1. **CSM Notes Parsing** (`ChatPromptTemplate → LLM → JsonOutputParser`):
   - Splits raw text on `---` dividers
   - Sends each note to LLM for structured extraction
   - Extracts: date, account name, sentiment, churn signals,
     expansion signals, competitors, stakeholders, action items

2. **Fuzzy Account Matching** (`fuzzywuzzy`):
   - Maps misspelled account names to canonical accounts
   - Uses `token_sort_ratio` for robust matching
   - Falls back to explicit ID matching when available
   - Example: `"BritePath Solutions"` → `"BrightPath Solutions"` (score=95)

3. **NPS Translation** (`ChatPromptTemplate → LLM → JsonOutputParser`):
   - Detects non-ASCII characters to identify non-English text
   - Translates Chinese, French, Spanish comments to English
   - Account 1017 (Pacific Rim Trading): Mandarin comment translated

4. **NPS Contradiction Detection** (`ChatPromptTemplate → LLM → JsonOutputParser`):
   - Identifies score-comment mismatches
   - Account 1019: Score=3 but comment says "Great developer experience"
   - Account 1041: Score=2 but comment says "Best headless CMS on the market"
   - Also flags generic/templated responses (same comment across 5+ accounts)

5. **Cross-Source Conflict Detection** (rule-based):
   - NPS score vs comment sentiment mismatch
   - NPS vs CSM sentiment disagreement
   - CSM positive notes vs heavy support ticket load

### `src/feature_engineering.py` — Signal Computation

Computes ~108 features per account organized into 7 groups:

**Usage Features (35 features):**
```
For each of {api_calls, content_entries_created, active_users, workflows_triggered}:
  - older_avg, recent_avg, pct_change, total, std, min, max, last, recent_slope

Derived flags:
  - api_calls_declining (>20% drop)
  - api_calls_severe_decline (>50% drop)
  - workflow_abandoned (had workflows, now zero)
  - usage_health_score (0-100 composite)
```

**Support Features (21 features):**
```
  - total_tickets, p1/p2/p3/p4 counts
  - open_tickets, escalated_tickets
  - open_p1_tickets (most important)
  - avg_resolution_hours, max_resolution_hours
  - recent_ticket_count (last 90 days)
  - ticket_velocity_trend
  - has_deprecation_tickets, has_migration_tickets
  - has_blocking_tickets, has_recurring_tickets
  - support_health_score (0-100)
```

**NPS Features (11 features):**
```
  - nps_score, nps_category (detractor/passive/promoter)
  - nps_is_detractor, nps_is_passive, nps_is_promoter
  - nps_is_contradictory, nps_is_generic, nps_is_non_english
  - nps_has_response (missing = disengagement signal)
  - nps_health_score (accounts for contradictions)
```

**CSM Features (16 features):**
```
  - csm_has_notes, csm_note_count
  - csm_sentiment (positive/negative/mixed/neutral)
  - csm_churn_signal_count, csm_expansion_signal_count
  - csm_competitor_count, csm_competitors_list
  - csm_has_executive_involvement
  - csm_has_missed_meetings
  - csm_has_budget_concerns
  - csm_has_evaluation_signals
  - csm_has_compliance_blockers
  - csm_has_migration_issues
  - csm_health_score (0-100)
```

**SDK Features (6 features):**
```
  - sdk_version, sdk_risk_level (critical/high/low/none)
  - sdk_health_score (critical=0, high=20, low=70, none=100)
  - sdk_is_critical, sdk_is_high_risk, sdk_is_v3
```

### `src/llm_pipeline.py` — Deep LLM Analysis (LangChain)

**Component 1: Changelog Impact Analysis**

First, extracts structured items from the changelog via LLM:
```
"SDK v3.x sunset April 30, 2026" → affects all accounts on sdk_version v3.x
"Legacy editor removal May 2026"  → affects accounts with editor complaints in notes
"Locale fallback fix in v4.2.3"   → affects accounts on v4.0/v4.1 with locale tickets
```

Then matches each item to accounts by:
- SDK version pattern matching
- Support ticket subject keyword matching
- CSM notes keyword matching

**Component 2: Silent Churn Detection**

Per-account LLM analysis combining ALL signals to detect 8 non-obvious patterns:
1. **Happy But Leaving**: High NPS + declining usage
2. **Building Alternatives**: Positive engagement + data export questions
3. **Gone Quiet**: No tickets, flat usage, no NPS response
4. **Survey Fatigue**: Generic NPS responses = disengaged relationship
5. **Platform Stagnation**: Never upgraded SDK = not invested in platform
6. **Champion Dependency**: 1-2 active users = fragile adoption
7. **Organizational Change**: Mergers, leadership changes
8. **Compliance Cliff**: Regulatory requirements vendor can't meet

**Component 3: Portfolio-Level Insights**

Analyzes the entire renewal portfolio for systemic patterns:
- CSM workload concentration risk
- Industry cluster vulnerabilities
- SDK sunset deadline bunching
- ARR-weighted risk concentration

### `src/risk_scoring.py` — LangGraph Pipeline

**State machine flow:**

```
Node 1: compute_quantitative_score
  Input:  7 health/risk scores
  Output: weighted_score × urgency_multiplier × arr_multiplier
  Logic:  Pure Python math, no LLM

Node 2: run_qualitative_assessment (LLM)
  Input:  quantitative score + all signals
  Output: adjustment (-15 to +15) + relationship_health + flags
  Logic:  LLM considers nuances numbers miss

Node 3: resolve_conflicts
  Input:  list of data source conflicts
  Output: adjustment (0 to ~20) + trust decisions
  Logic:  Rule-based trust hierarchy:
          Support tickets > CSM notes > NPS score > NPS comment

Node 4: assign_final_risk
  Input:  quant + qual_adj + conflict_adj
  Output: final_score + risk_tier + recommended_actions
  Logic:  Score = sum; override rules for compliance/competitor/SDK

Node 5: calibrate_confidence
  Input:  data completeness + conflict count
  Output: high/medium/low confidence label
  Logic:  Penalizes missing data, contradictions, generic NPS
```

**Weight distribution:**
```
Usage trends:     25%  — Strongest behavioral predictor
CSM sentiment:    25%  — Rich qualitative context
Support tickets:  15%  — Real product experience
NPS score:        10%  — Useful but often unreliable
SDK platform:     10%  — Technical deadline pressure
Changelog:         8%  — Product-driven risk
Silent churn:      7%  — Non-obvious pattern detection
```

### `src/explanation_generator.py` — Narrative Generation (LangChain)

Generates three types of output:

1. **Per-Account Narratives** (High + Medium risk only):
   - Written as if briefing VP of Customer Success
   - 4-section structure: Risk Summary → Key Signals → Missing Data → Next Steps
   - Specific, time-bound action items (not generic advice)

2. **Executive Summary**:
   - Portfolio overview with ARR at risk
   - CSM workload analysis
   - Non-obvious insights
   - This week vs this month priorities

3. **CSM Briefings**:
   - One per CSM who has at-risk accounts
   - This week's priorities (top 2-3 actions)
   - Account-by-account status (one line each)
   - Escalation needs

---

## 11. Data Sources & Inconsistencies Handled

### Inconsistencies Detected and Resolved

| Issue | Raw Data | Resolved To | Method |
|-------|----------|-------------|--------|
| Typo in account name | `"BritePath Solutions"` | `"BrightPath Solutions"` | FuzzyWuzzy (score=95) |
| Typo in account name | `"Pinacle Media"` | `"Pinnacle Media Group"` | FuzzyWuzzy (score=87) |
| Typo in account name | `"Thunderbolt Moters"` | `"Thunderbolt Motors"` | FuzzyWuzzy (score=91) |
| Wrong CSM in notes | Note says "James O." for NovaTech but accounts.csv says "Sarah Chen" | Matched by account name, noted discrepancy | Fuzzy match + flag |
| Ambiguous note ref | Note says "Harbourside Dining" for account 1099 which is "Oakridge Retail" | Linked via context (Emily W.) | LLM + ID reference |
| Non-English NPS | Account 1017: Mandarin comment | Translated: "Product functions are acceptable, but customer service team communication efficiency is too low..." | LangChain translation |
| Non-English NPS | Account 1013: Spanish comment | Translated: "The product is good but support in Spanish is non-existent" | LangChain translation |
| Non-English NPS | Account 1014/1016: French comment | Translated: "The product is good but the interface is not intuitive for our marketing team" | LangChain translation |
| Score-comment contradiction | Account 1019: NPS=3, comment says "Great developer experience" | Flagged as contradictory, trust lower score | LLM detection |
| Score-comment contradiction | Account 1041: NPS=2, comment says "Best headless CMS on the market" | Flagged, generic template discounted | LLM detection |
| Generic/templated NPS | "Best headless CMS on the market" used by 12+ accounts | Flagged as unreliable signal | Frequency analysis |
| Silent churn pattern | Meridian Health: NPS=8 but building homegrown replacement | Flagged as high silent churn risk | LLM pattern detection |
| Missing NPS | 22 accounts have no NPS response | Treated as mild disengagement signal | Feature flag |

### Changelog → Account Mapping

The changelog is used to identify accounts at specific technical risk:

| Changelog Item | Accounts Affected | Risk Added |
|----------------|------------------|------------|
| SDK v3.x sunset (Apr 30, 2026) | All accounts on `v3.x` | Critical SDK risk |
| Legacy editor removal (May 2026) | Accounts with editor complaints | High platform risk |
| Locale fallback fix in v4.2.3 | Accounts on v4.0/v4.1 with locale tickets | Medium technical debt |
| Workflow engine deprecated | Accounts with workflow usage decline | Medium |
| REST API v2 sunset (Apr 30, 2026) | Accounts with deprecation tickets | Critical |

---

## 12. Risk Scoring Methodology

### Score Composition

```
final_risk_score = quantitative_score + qualitative_adjustment + conflict_adjustment
                     (0-100)              (-15 to +15)              (0 to ~20)
```

### Quantitative Score

```python
weighted_health = (
    0.25 × usage_health_score +
    0.25 × csm_health_score +
    0.15 × support_health_score +
    0.10 × nps_health_score +
    0.10 × sdk_health_score +
    0.08 × (100 - changelog_risk_score) +
    0.07 × (100 - silent_churn_score)
)

raw_risk = 100 - weighted_health
adjusted_risk = raw_risk × urgency_multiplier × arr_multiplier
```

**Urgency multipliers:**
- Renewal in ≤15 days: ×1.3
- Renewal in ≤30 days: ×1.2
- Renewal in ≤45 days: ×1.1
- Renewal in ≤90 days: ×1.0

**ARR multipliers:**
- ARR ≥ $1M: ×1.15 (Enterprise accounts get conservative scoring)
- ARR ≥ $500K: ×1.10
- ARR ≥ $100K: ×1.05
- ARR < $100K: ×1.0

### Qualitative Adjustment (LLM)

The LLM analyzes nuances that numbers miss:
- Active competitor POC underway → +10 to +15
- Organizational change (merger/acquisition) → +5 to +10
- Relationship breakdown (CSM changed requests) → +5 to +10
- Strong expansion signals despite some risk → -5 to -10

### Conflict Resolution Adjustment

When data sources disagree, we add risk (conservative principle):
- NPS score contradicts comment: +5
- NPS vs CSM sentiment disagree: +3 to +8
- CSM positive vs support tickets heavy: +5
- Generic NPS with extreme score: +3

### Override Rules (Force High)

Certain conditions force High tier regardless of score:
1. Actively evaluating competitors AND CSM sentiment is negative
2. On deprecated SDK v3.x AND renewal within 30 days
3. Compliance blocker exists (security questionnaire, SOC 2, GDPR)

### Risk Tiers

| Tier | Score Range | Meaning |
|------|-------------|---------|
| 🔴 High | ≥ 70 | Immediate action required. High probability of churn/downgrade. |
| 🟡 Medium | 40-69 | Proactive attention needed. Risk signals present but recoverable. |
| 🟢 Low | < 40 | Standard renewal process. Monitor for changes. |

---

## 13. LangChain & LangGraph Usage

### Where LangChain Is Used

| Location | Chain Type | Purpose |
|----------|-----------|---------|
| `data_reconciliation.py` | `ChatPromptTemplate → ChatGroq → JsonOutputParser` | Parse messy CSM notes into structured JSON |
| `data_reconciliation.py` | `ChatPromptTemplate → ChatGroq → JsonOutputParser` | Translate non-English NPS comments |
| `data_reconciliation.py` | `ChatPromptTemplate → ChatGroq → JsonOutputParser` | Detect NPS score-comment contradictions |
| `llm_pipeline.py` | `ChatPromptTemplate → ChatGroq → JsonOutputParser` | Extract structured items from changelog |
| `llm_pipeline.py` | `ChatPromptTemplate → ChatGroq → JsonOutputParser` | Detect silent churn patterns per account |
| `llm_pipeline.py` | `ChatPromptTemplate → ChatGroq → JsonOutputParser` | Generate portfolio-level insights |
| `risk_scoring.py` | `ChatPromptTemplate → ChatGroq → JsonOutputParser` | Qualitative risk adjustment per account |
| `explanation_generator.py` | `ChatPromptTemplate → ChatGroq → StrOutputParser` | Generate account risk narratives |
| `explanation_generator.py` | `ChatPromptTemplate → ChatGroq → StrOutputParser` | Generate executive summary |
| `explanation_generator.py` | `ChatPromptTemplate → ChatGroq → StrOutputParser` | Generate CSM briefings |

### Where LangGraph Is Used

The entire risk scoring pipeline in `risk_scoring.py` is a LangGraph
`StateGraph` with 5 nodes and typed state:

```python
# State flows through nodes sequentially
workflow = StateGraph(AccountRiskState)
workflow.add_node("quantitative_score", compute_quantitative_score)
workflow.add_node("qualitative_assessment", run_qualitative_assessment)
workflow.add_node("conflict_resolution", resolve_conflicts)
workflow.add_node("final_risk", assign_final_risk)
workflow.add_node("confidence", calibrate_confidence)
```

**Why LangGraph for this?**

- **Typed state** (`AccountRiskState` TypedDict) ensures data integrity
  across the pipeline — no silent field drops
- **Node isolation** makes each step independently testable and debuggable
- **Sequential guarantees** — qualitative assessment always runs after
  quantitative, ensuring the LLM has the computed score as context
- **Future-proof** — easy to add conditional edges (e.g., "if compliance
  blocker, skip to force_high node") without restructuring the code
- **State accumulation** — each node adds to state rather than replacing
  it, so the final output contains the complete reasoning chain

---

## 14. Non-Obvious Insights

This is the most important differentiator from rule-based systems.
Here are the specific non-obvious patterns the engine detects:

### Pattern 1: Silent Churn (High NPS + Declining Usage)

**Example:** Meridian Health (Account 1003)
- NPS score: 8 (looks healthy)
- CSM notes: "Good news/bad news... actual usage has cratered"
- Reality: They built a custom middleware layer and are migrating away
- Why rule-based misses it: NPS ≥ 7 would be green-flagged

**How we catch it:**
- Feature engineering detects usage decline
- CSM notes parsing extracts "migrating content" signal
- Silent churn LLM analysis identifies "HAPPY BUT LEAVING" pattern
- NPS vs CSM conflict detection flags the discrepancy

### Pattern 2: Survey Fatigue as Disengagement Signal

**Example:** Multiple accounts with NPS score 2-4 but comment
"Best headless CMS on the market. Period."

- The positive comment is copy-pasted boilerplate
- The LOW SCORE (2-4) is the real signal
- Why rule-based misses it: Most systems look at the comment text

**How we catch it:**
- Frequency analysis: comments appearing 5+ times are flagged generic
- Low score + generic positive comment = contradiction flag
- NPS health score calculation discounts generic templates

### Pattern 3: Non-English Comment Hides Critical Feedback

**Example:** Pacific Rim Trading (Account 1017)
- NPS score: 5 (passive)
- Comment: `产品功能还可以，但客服团队沟通效率太低了...`
- CSM note: "their last NPS comment was in mandarin so I couldn't read it"
- Translation: "Product functions are acceptable, but customer service
  communication efficiency is too low. We have repeatedly requested a
  new account manager but got no response. Very disappointed."

**How we catch it:**
- Non-ASCII character detection identifies non-English text
- LangChain translation with sentiment extraction
- Result: Score=5 + very negative sentiment → elevated risk

### Pattern 4: Executive Escalation Pattern

**Example:** Zenith Publishing (Account 1007)
- CRO cc'd on email thread
- 30% discount demand
- Competitor POC already underway

The CRO involvement + competitor POC combination is a near-certain
churn signal that requires C-level response, not a CSM call.

**How we catch it:**
- CSM notes parsing extracts stakeholder titles
- `csm_has_executive_involvement` flag triggers LLM qualitative escalation
- Override rules force executive_attention_needed flag

### Pattern 5: SDK Deadline Cluster Risk

**Example:** Multiple v3.x accounts all hitting the April 30, 2026 deadline.

Individually, each account's SDK migration is manageable.
But if 8 accounts all need engineering migration in the same 2-week
window, the SA team is overwhelmed, migrations fail, and accounts churn.

**How we catch it:**
- SDK risk assessment per account
- Portfolio insights LLM identifies the cluster pattern
- Quantifies combined ARR at risk from the deadline cluster

### Pattern 6: Compliance Cliff

**Example:** Atlas Financial (Account 1006)
- Regulated industry (Government)
- Asked about SOC 2 audit timeline
- Single-tenancy requirement not on their current plan
- SDK v3.1 hitting deprecated endpoints daily

This is forced churn — not about product satisfaction at all.
No amount of feature improvement fixes a compliance gap.

**How we catch it:**
- CSM notes parsing extracts compliance/security signals
- `csm_has_compliance_blockers` flag
- Override rule: compliance blocker → force High tier

---

## 15. Output Files

All output files are saved to the `outputs/` directory.

### `risk_scored_accounts.csv`

Main output. One row per renewal account.

| Column | Description |
|--------|-------------|
| `account_id` | Account ID |
| `account_name` | Account name |
| `arr` | Annual recurring revenue |
| `plan_tier` | Starter/Growth/Scale/Enterprise |
| `industry` | Industry vertical |
| `region` | Geographic region |
| `csm_name` | Assigned CSM |
| `contract_end_date` | Contract expiration |
| `days_until_renewal` | Days from reference date |
| `final_risk_score` | 0-100 (higher = more risky) |
| `risk_tier` | High/Medium/Low |
| `confidence` | high/medium/low |
| `quantitative_score` | Pre-adjustment score |
| `qualitative_adjustment` | LLM adjustment (-15 to +15) |
| `conflict_adjustment` | Conflict resolution adjustment |
| `usage_health` | 0-100 |
| `support_health` | 0-100 |
| `nps_health` | 0-100 |
| `csm_health` | 0-100 |
| `sdk_health` | 0-100 |
| `relationship_health` | strong/stable/strained/critical |
| `executive_attention` | Boolean |
| `risk_factors` | JSON array of risk factors |
| `recommended_actions` | JSON array of actions |
| `competitors` | Competitor products mentioned |
| `conflict_notes` | Data source conflict explanations |
| `silent_churn_patterns` | JSON array of detected patterns |

### `executive_summary.md`
Portfolio-level briefing for CRO/VP CS. Markdown formatted.
Includes: risk tier breakdown, top 5 accounts, CSM workload,
portfolio insights, and top priority action.

### `account_narratives.md`
Per-account risk narratives for all High and Medium risk accounts.
Written as internal CS briefings. Structured as:
Risk Summary → Key Signals → Data Gaps → Recommended Actions.

### `csm_briefings.md`
One section per CSM who has at-risk renewal accounts.
Contains: this week's priorities, account status, escalation needs.

### `changelog_impacts.csv`
Which product changes affect which renewal accounts.
Includes changelog_risk_score per account.

### `silent_churn_analysis.csv`
Silent churn patterns detected per account.
Includes pattern names, confidence levels, evidence, and risk scores.

### `portfolio_insights.json`
Non-obvious portfolio-level insights in structured JSON.
Includes: insight title, detail, evidence, ARR at risk, recommended
action, urgency, and insight type.

---

## 16. Tradeoffs & Design Decisions

### Decision 1: Groq over GPT-4

**Choice:** Groq (llama-3.1-8b-instant) as primary, OpenRouter as fallback

**Why:** Groq's free tier provides 14,400 requests/day at very low latency
(~100-200ms per call). For a pipeline making 50-80 LLM calls, this is
crucial. GPT-4 would cost $5-15 per pipeline run and take 2-3× longer.

**Tradeoff:** llama-3.1-8b occasionally produces malformed JSON or
misses subtle nuances. Mitigated with JsonOutputParser + retry logic
with exponential backoff.

### Decision 2: Reference Date = April 15, 2026

**Why:** The latest CSM notes are from April 10, 2026. Using April 15
makes the "next 90 days" window realistic given the data.

**Tradeoff:** Makes the tool feel like a snapshot rather than live.
In production, `REFERENCE_DATE` would be `date.today()`.

### Decision 3: Conservative Signal Trust Hierarchy

**Why:** When NPS disagrees with CSM notes, we trust CSM notes.
When CSM notes disagree with support tickets, we trust tickets.

**Rationale:** Data that requires customer action (filing a ticket,
speaking on a call) is harder to fabricate or misremember than
periodic surveys.

**Tradeoff:** Could over-flag healthy accounts with active support usage.
Mitigated by the confidence calibration node.

### Decision 4: Health Score Abstraction

**Why:** Each data source produces one 0-100 "health score" before
being fed into the weighted average.

**Benefits:**
- Interpretable (everyone understands a 0-100 scale)
- Comparable across very different data types
- Easy to adjust weights without rewriting scoring logic

**Tradeoff:** Information loss — the raw features contain more nuance
than the aggregate score. Addressed by keeping raw features available
for LLM qualitative analysis.

### Decision 5: LangGraph for Risk Scoring

**Why:** Could have been a simple function chain. Used LangGraph because:
- Typed state catches bugs early
- Each node is independently testable
- Easy to add conditional branches later
- Demonstrates architectural thinking for production

**Tradeoff:** More boilerplate than a simple pipeline. Justified for
a system that will evolve over time.

### Decision 6: Force-High Override Rules

**Why:** Some signals are "dealbreakers" regardless of aggregate score.
A compliance blocker is binary — either you can meet it or you can't.

**Examples of override conditions:**
- Compliance blocker exists
- Actively in competitor evaluation AND negative CSM sentiment
- Deprecated SDK with renewal within 30 days

**Tradeoff:** Reduces the value of the score for these accounts.
Mitigated by showing the override reason in the output.

---

## 17. What I'd Do With More Time

### 1. Historical Training Data for ML Model
Replace the weighted average with a proper ML model (Random Forest
or XGBoost) trained on historical renewal outcomes. The feature
matrix (~108 features) is already designed for this. Would need:
- 2-3 years of historical renewal outcomes
- SHAP values for explainability

### 2. Real-Time Pipeline with Incremental Updates
The current pipeline is a batch process. Production needs:
- Webhook triggers when new support tickets are filed
- Daily usage metric ingestion
- Event-driven re-scoring when signals change

### 3. Salesforce/CRM Integration
Pull account data directly from Salesforce instead of CSV files.
Use `simple-salesforce` library for real-time data.

### 4. LLM Fine-Tuning on Historical CS Notes
Fine-tune a smaller model (llama-3.1-8b) on historical CSM notes
to be better at extracting CS-specific risk signals. Would dramatically
improve accuracy of CSM notes parsing.

### 5. Confidence Intervals on Risk Scores
Instead of point estimates, show ranges: "70-85 risk (high confidence)"
vs "40-75 risk (low confidence)". Would require Monte Carlo simulation
or Bayesian approaches.

### 6. Champion Tracking
Add a data source for contact-level engagement (email open rates,
meeting attendance). "Champion leaving the company" is the highest-
signal churn predictor that's missing from current data.

### 7. Multi-Language CSM Notes Support
CSM notes are assumed to be in English. Some international accounts
(LATAM, EMEA) likely have notes in Spanish, French, German. Add
translation as a preprocessing step.

### 8. A/B Testing Framework for Interventions
Track which recommended actions lead to successful renewals.
Feed outcomes back into the scoring model to improve recommendations.

---

## 18. What I'd Change for Production

### Architecture Changes

1. **Replace CSVs with a Data Warehouse**
   - Source data from Snowflake/BigQuery
   - Scheduled daily ingestion via Airflow or Prefect
   - Store computed features and scores in a separate mart table

2. **Replace In-Memory Processing with Stream Processing**
   - Apache Kafka for real-time ticket/usage event ingestion
   - Flink or Spark Streaming for continuous feature computation
   - Score recalculated when key signals change

3. **Add a Vector Database for CSM Notes**
   - Store CSM notes as embeddings in Pinecone or Weaviate
   - Semantic search: "find all accounts that mentioned Hygraph"
   - Similarity matching: "find accounts with patterns similar to churned accounts"

4. **LLM Cost Optimization**
   - Cache LLM responses for unchanged inputs
   - Use smaller models for routine tasks (translation, sentiment)
   - Reserve large models (GPT-4, llama-70b) for final narrative generation
   - Estimated production cost: ~$50-200/month with caching

5. **Rate Limiting and Circuit Breakers**
   - Implement token bucket rate limiting for Groq API
   - Circuit breaker pattern for API outages
   - Fallback to rule-based scoring when LLM unavailable

### Code Quality

6. **Observability**
   - OpenTelemetry tracing across the pipeline
   - Log all LLM inputs/outputs for debugging
   - Track model drift: are LLM-generated scores consistent over time?

7. **Testing**
   - Unit tests for all feature computation functions
   - Snapshot tests for LLM prompts (detect prompt drift)
   - Integration tests with mock LLM responses
   - Property-based tests for edge cases (zero usage, missing NPS, etc.)

8. **Configuration Management**
   - Risk weights configurable via YAML (not hardcoded)
   - Tier thresholds adjustable without code changes
   - Ability to add new signals without changing core scoring logic

### Operational

9. **Human-in-the-Loop Review**
   - CSMs can flag scores as "disagree" → feeds back to model
   - Periodic calibration sessions with CS leadership
   - "Why did this account get flagged?" audit trail

10. **Privacy and Compliance**
    - PII handling for contact names, email addresses
    - Data retention policies for LLM-processed content
    - Audit log for who accessed which account risk profile
    - GDPR compliance for EU customer data

---

## 19. Troubleshooting

### "No module named 'langchain_groq'"

```bash
pip install langchain-groq langchain-openai
```

### "GROQ_API_KEY not set"

Make sure your `.env` file exists and has the correct key:

```bash
cat .env
# Should show: GROQ_API_KEY=gsk_xxxx
```

Also make sure you're running from the project root:

```bash
# You should be in: /path/to/renewal-intelligence-engine/
python cli.py run
```

### "Account X not found in renewal window"

The account exists but its contract end date is outside the 90-day window.
Check the account's renewal date:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/accounts.csv')
print(df[df['account_id'] == X][['account_name', 'contract_end_date']])
"
```

### LLM API Rate Limiting

Groq's free tier allows 30 requests/minute. If you see rate limit errors:

1. The pipeline has built-in `time.sleep(0.5)` between calls
2. If still failing, increase the sleep time in `llm_pipeline.py`
3. Or switch to OpenRouter: `python cli.py run --provider openrouter`

### "JsonOutputParser failed"

The LLM returned malformed JSON. The pipeline has retry logic (3 attempts).
If it still fails:

1. Check if your Groq API key is valid
2. Try a different model in `config.py`:
   ```python
   GROQ_MODEL = "llama-3.3-70b-versatile"  # More capable, slower
   ```

### Streamlit "No results found"

Run the pipeline first:
```bash
python cli.py run
```

Then refresh the Streamlit page.

### Streamlit shows old results after re-running pipeline

Clear Streamlit cache:
```bash
# In the browser, go to the top-right menu → Clear Cache
# Or restart Streamlit:
streamlit run app.py
```

### "FuzzyWuzzy warning about python-Levenshtein"

This is just a performance warning, not an error:
```bash
pip install python-Levenshtein
```

---

## 20. API Keys & Cost Estimates

### Groq (Recommended)

- **Cost:** Free tier — 14,400 requests/day, 30 requests/minute
- **Model:** `llama-3.1-8b-instant`
- **Latency:** 100-200ms per call
- **Sign up:** https://console.groq.com
- **Usage per pipeline run:** ~55-80 API calls

At 80 calls per run and 30 calls/minute, the pipeline takes ~3 minutes
of API time (plus processing). Well within free limits.

### OpenRouter

- **Cost:** Free tier available for some models
- **Model:** `meta-llama/llama-3.1-70b-instruct:free`
- **Use case:** More complex analysis when Groq's 8B model isn't sufficient
- **Sign up:** https://openrouter.ai

### Estimated Total Cost

| Scenario | Cost |
|----------|------|
| Development (Groq free tier) | $0 |
| Production daily batch (Groq) | $0 (within free limits) |
| Production with GPT-4 | ~$5-15 per pipeline run |
| Production with caching | <$50/month |

---

## License

Internal tool — not for distribution.

---

## Contributors

Built as a take-home assignment for the Applied AI Engineer role at
Contentstack BizOps team.

---

*Generated by Renewal Intelligence Engine v1.0*
*Reference Date: 2026-04-15 | Renewal Window: 90 days*
```

---

## `.env.example` — Template for Setup

```env
# Renewal Intelligence Engine — Environment Variables
# Copy this file to .env and fill in your API keys

# Required: Groq API Key (free at console.groq.com)
GROQ_API_KEY=gsk_your_groq_api_key_here

# Optional: OpenRouter API Key (free tier at openrouter.ai)
OPENROUTER_API_KEY=sk-or-v1-your_openrouter_key_here

# Reference date for the pipeline (ISO format: YYYY-MM-DD)
# Leave as 2026-04-15 for the synthetic dataset
# Change to today's date for production use
REFERENCE_DATE=2026-04-15
```

---

## Final Checklist Before Running

```bash
# 1. Verify directory structure
ls data/
# Should show: accounts.csv usage_metrics.csv support_tickets.csv
#              csm_notes.txt nps_responses.csv changelog.md

# 2. Verify .env
cat .env
# Should show GROQ_API_KEY=gsk_xxxx

# 3. Verify dependencies
python -c "import langchain; import langgraph; import streamlit; print('All OK')"

# 4. Run quick validation (no LLM calls)
python test_part1.py

# 5. Run full pipeline
python cli.py run

# 6. Verify outputs
ls outputs/
# Should show: risk_scored_accounts.csv executive_summary.md etc.

# 7. Launch dashboard
streamlit run app.py
```