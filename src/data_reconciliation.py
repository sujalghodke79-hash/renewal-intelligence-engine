"""
Data Reconciliation Module

Handles:
1. CSM notes parsing — extract structured data from messy text
2. Fuzzy matching — account names in notes vs accounts.csv
3. NPS anomaly detection — score vs sentiment contradictions
4. Non-English NPS translation
5. Cross-source conflict identification
6. SDK version risk flagging from changelog
"""

import re
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from fuzzywuzzy import fuzz, process

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

from src.config import (
    REFERENCE_DATE, RENEWAL_CUTOFF_DATE,
    SDK_V3_SUNSET_DATE, REST_API_V2_SUNSET
)
from src.utils import get_llm


# ============================================================
# 1. CSM NOTES PARSING
# ============================================================

class CSMNoteEntry(BaseModel):
    """Structured representation of a single CSM note."""
    date: str = Field(description="Date of the note in YYYY-MM-DD format, best guess if ambiguous")
    account_name_raw: str = Field(description="Account name exactly as written in the note (may have typos)")
    account_id_if_mentioned: Optional[str] = Field(description="Account ID if explicitly mentioned, else null")
    csm_name_raw: Optional[str] = Field(description="CSM name if mentioned, else null")
    sentiment: str = Field(description="Overall sentiment: positive, negative, neutral, or mixed")
    churn_risk_signals: List[str] = Field(description="List of specific churn/downgrade risk signals mentioned")
    expansion_signals: List[str] = Field(description="List of expansion or positive signals mentioned")
    competitors_mentioned: List[str] = Field(description="Names of competitor products mentioned")
    key_stakeholders: List[str] = Field(description="Titles/roles of people mentioned on the call")
    action_items: List[str] = Field(description="Any action items or follow-ups mentioned")
    summary: str = Field(description="One-paragraph summary of the note")


def split_csm_notes(raw_text: str) -> List[str]:
    """
    Split raw CSM notes text into individual note entries.
    Notes are separated by '---' lines.
    """
    # Split on --- dividers
    sections = re.split(r'\n\s*---+\s*\n', raw_text)

    # Clean and filter empty sections
    notes = []
    for section in sections:
        cleaned = section.strip()
        if cleaned and len(cleaned) > 20:  # Skip very short fragments
            # Remove the header if it's the first section
            if cleaned.startswith("=== CSM Call Notes"):
                # Extract everything after the header line
                lines = cleaned.split('\n')
                # Find where actual notes start (skip header lines)
                start_idx = 0
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith('===') and not line.startswith('(Internal'):
                        start_idx = i
                        break
                cleaned = '\n'.join(lines[start_idx:]).strip()
                if len(cleaned) < 20:
                    continue
            notes.append(cleaned)

    print(f"[Reconciliation] Split CSM notes into {len(notes)} individual entries")
    return notes


def parse_csm_notes_with_llm(
        notes: List[str],
        provider: str = "groq"
) -> List[Dict[str, Any]]:
    """
    Use LLM to extract structured data from each CSM note.
    """
    llm = get_llm(provider=provider, temperature=0.0, max_tokens=3000)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at parsing messy CRM/CSM call notes into structured data.
You will receive a single CSM note entry. Extract the information precisely.

IMPORTANT RULES:
- Account names may have typos — extract them EXACTLY as written
- Dates come in many formats — normalize to YYYY-MM-DD (assume year 2026 if only month/day given, unless context says otherwise)
- For March dates use 2026-03-XX, for April dates use 2026-04-XX
- Be exhaustive about risk signals — competitor mentions, budget cuts, evaluation of alternatives, missed meetings, compliance blockers, etc.
- Be exhaustive about expansion signals — seat additions, new use cases, product love, etc.
- Sentiment should reflect RENEWAL RISK, not just tone of note

Return valid JSON only. No markdown, no explanation."""),
        ("human", """Parse this CSM note into structured JSON with these fields:
- date (string, YYYY-MM-DD)
- account_name_raw (string, exactly as written)
- account_id_if_mentioned (string or null)
- csm_name_raw (string or null)  
- sentiment (string: positive/negative/neutral/mixed)
- churn_risk_signals (list of strings)
- expansion_signals (list of strings)
- competitors_mentioned (list of strings)
- key_stakeholders (list of strings - titles/roles)
- action_items (list of strings)
- summary (string - one paragraph)

CSM NOTE:
{note}""")
    ])

    chain = prompt | llm | JsonOutputParser()

    parsed_notes = []
    for i, note in enumerate(notes):
        try:
            print(f"  Parsing note {i + 1}/{len(notes)}...")
            result = chain.invoke({"note": note})
            result['_raw_text'] = note
            result['_note_index'] = i
            parsed_notes.append(result)
        except Exception as e:
            print(f"  WARNING: Failed to parse note {i + 1}: {e}")
            # Store raw note with minimal structure for manual review
            parsed_notes.append({
                'date': 'unknown',
                'account_name_raw': 'PARSE_FAILED',
                'account_id_if_mentioned': None,
                'csm_name_raw': None,
                'sentiment': 'unknown',
                'churn_risk_signals': [],
                'expansion_signals': [],
                'competitors_mentioned': [],
                'key_stakeholders': [],
                'action_items': [],
                'summary': f'PARSE FAILED: {str(e)}',
                '_raw_text': note,
                '_note_index': i
            })

    print(f"[Reconciliation] Successfully parsed {len(parsed_notes)} CSM notes")
    return parsed_notes


# ============================================================
# 2. FUZZY MATCHING — CSM Notes to Accounts
# ============================================================

def build_account_lookup(accounts_df: pd.DataFrame) -> Dict:
    """Build lookup dictionaries for account matching."""
    lookup = {
        'by_id': {},
        'by_name': {},
        'name_list': [],
        'id_to_name': {},
        'name_to_id': {}
    }

    for _, row in accounts_df.iterrows():
        aid = str(row['account_id'])
        name = row['account_name']

        lookup['by_id'][aid] = row.to_dict()
        lookup['by_name'][name.lower()] = row.to_dict()
        lookup['name_list'].append(name)
        lookup['id_to_name'][aid] = name
        lookup['name_to_id'][name.lower()] = aid

    return lookup


def match_note_to_account(
        parsed_note: Dict,
        account_lookup: Dict,
        threshold: int = 70
) -> Dict:
    """
    Match a parsed CSM note to an account using:
    1. Explicit account ID if mentioned
    2. Fuzzy name matching

    Returns the parsed note with added fields:
    - matched_account_id
    - matched_account_name
    - match_confidence
    - match_method
    """
    note = parsed_note.copy()

    # Method 1: Direct ID match
    raw_id = note.get('account_id_if_mentioned')
    if raw_id:
        # Clean the ID — extract just digits
        clean_id = re.sub(r'[^0-9]', '', str(raw_id))
        if clean_id in account_lookup['by_id']:
            note['matched_account_id'] = int(clean_id)
            note['matched_account_name'] = account_lookup['id_to_name'][clean_id]
            note['match_confidence'] = 100
            note['match_method'] = 'explicit_id'
            return note

    # Method 2: Fuzzy name matching
    raw_name = note.get('account_name_raw', '')
    if raw_name and raw_name != 'PARSE_FAILED':
        # Try exact match first (case-insensitive)
        if raw_name.lower() in account_lookup['name_to_id']:
            aid = account_lookup['name_to_id'][raw_name.lower()]
            note['matched_account_id'] = int(aid)
            note['matched_account_name'] = account_lookup['id_to_name'][aid]
            note['match_confidence'] = 100
            note['match_method'] = 'exact_name'
            return note

        # Fuzzy match
        best_match, score = process.extractOne(
            raw_name,
            account_lookup['name_list'],
            scorer=fuzz.token_sort_ratio
        )

        if score >= threshold:
            aid = account_lookup['name_to_id'][best_match.lower()]
            note['matched_account_id'] = int(aid)
            note['matched_account_name'] = best_match
            note['match_confidence'] = score
            note['match_method'] = f'fuzzy_match (score={score})'

            if raw_name.lower() != best_match.lower():
                print(f"    Fuzzy matched: '{raw_name}' → '{best_match}' (score={score})")

            return note

    # No match found
    note['matched_account_id'] = None
    note['matched_account_name'] = None
    note['match_confidence'] = 0
    note['match_method'] = 'no_match'
    print(f"    WARNING: Could not match note for '{raw_name}' (ID: {raw_id})")

    return note


# ============================================================
# 3. NPS ANOMALY DETECTION & TRANSLATION
# ============================================================

def translate_and_analyze_nps(
        nps_df: pd.DataFrame,
        provider: str = "groq"
) -> pd.DataFrame:
    """
    Process NPS responses:
    1. Translate non-English comments
    2. Detect score-vs-comment contradictions
    3. Flag templated/generic responses
    """
    llm = get_llm(provider=provider, temperature=0.0, max_tokens=1500)

    nps = nps_df.copy()
    nps['translated_comment'] = nps['verbatim_comment']
    nps['detected_language'] = 'en'
    nps['comment_sentiment_score'] = None  # -1 to 1 scale
    nps['is_contradictory'] = False
    nps['is_generic_template'] = False
    nps['contradiction_note'] = ''

    # Identify non-English comments
    non_ascii_mask = nps['verbatim_comment'].str.contains(
        r'[^\x00-\x7F]', regex=True, na=False
    )
    non_english_rows = nps[non_ascii_mask & (nps['verbatim_comment'] != '')]

    if len(non_english_rows) > 0:
        print(f"[Reconciliation] Translating {len(non_english_rows)} non-English NPS comments...")

        translate_prompt = ChatPromptTemplate.from_messages([
            ("system", "You translate text to English and analyze sentiment. Return valid JSON only."),
            ("human", """Translate this NPS comment to English and analyze it.

Comment: {comment}
NPS Score given: {score}

Return JSON:
{{
    "original_language": "language name",
    "translated_text": "English translation",
    "sentiment_score": float between -1.0 (very negative) and 1.0 (very positive),
    "sentiment_summary": "one sentence summary of the sentiment"
}}""")
        ])

        translate_chain = translate_prompt | llm | JsonOutputParser()

        for idx in non_english_rows.index:
            comment = nps.at[idx, 'verbatim_comment']
            score = nps.at[idx, 'score']
            try:
                result = translate_chain.invoke({
                    "comment": comment,
                    "score": score
                })
                nps.at[idx, 'translated_comment'] = result.get('translated_text', comment)
                nps.at[idx, 'detected_language'] = result.get('original_language', 'unknown')
                nps.at[idx, 'comment_sentiment_score'] = result.get('sentiment_score', 0)
                print(f"    Account {nps.at[idx, 'account_id']}: {result.get('original_language')} → English")
            except Exception as e:
                print(f"    WARNING: Translation failed for account {nps.at[idx, 'account_id']}: {e}")

    # Detect GENERIC/TEMPLATED comments
    # These are exact duplicates used across multiple accounts
    comment_counts = nps['verbatim_comment'].value_counts()
    generic_comments = comment_counts[comment_counts >= 3].index.tolist()
    generic_comments = [c for c in generic_comments if c != '']  # Exclude empty

    if generic_comments:
        print(f"[Reconciliation] Found {len(generic_comments)} templated/generic comments:")
        for gc in generic_comments:
            count = comment_counts[gc]
            print(f"    '{gc[:60]}...' — used by {count} accounts")

        nps.loc[nps['verbatim_comment'].isin(generic_comments), 'is_generic_template'] = True

    # Detect SCORE-COMMENT CONTRADICTIONS
    # High score (9-10) + negative sentiment OR Low score (0-4) + positive sentiment
    print("[Reconciliation] Checking for score-comment contradictions...")

    contradiction_prompt = ChatPromptTemplate.from_messages([
        ("system", """You detect contradictions between NPS scores and comments.
An NPS score of 0-6 is detractor, 7-8 is passive, 9-10 is promoter.
Return valid JSON only."""),
        ("human", """Analyze if this NPS response has a contradiction between the score and comment.

NPS Score: {score}  
Comment: {comment}

Return JSON:
{{
    "comment_sentiment": "positive" or "negative" or "neutral",
    "sentiment_score": float from -1.0 to 1.0,
    "is_contradictory": boolean,
    "explanation": "why it is or isn't contradictory"
}}""")
    ])

    contradiction_chain = contradiction_prompt | llm | JsonOutputParser()

    # Only check non-empty, non-generic comments
    check_mask = (
            (nps['verbatim_comment'] != '') &
            (~nps['is_generic_template']) &
            (nps['detected_language'] == 'en')  # Already translated non-English above
    )
    rows_to_check = nps[check_mask]

    for idx in rows_to_check.index:
        comment = nps.at[idx, 'translated_comment']
        score = nps.at[idx, 'score']

        # Quick heuristic pre-filter: only LLM-check if there's potential mismatch
        is_detractor = score <= 6
        is_promoter = score >= 9
        has_positive_words = any(
            w in comment.lower() for w in ['love', 'great', 'best', 'transformed', 'phenomenal', 'recommend'])
        has_negative_words = any(w in comment.lower() for w in
                                 ['done', 'frustrated', 'downgrade', 'waste', 'forever', 'cliff', 'disappointed'])

        potential_contradiction = (is_detractor and has_positive_words) or (is_promoter and has_negative_words)

        if potential_contradiction:
            try:
                result = contradiction_chain.invoke({
                    "score": score,
                    "comment": comment
                })
                nps.at[idx, 'comment_sentiment_score'] = result.get('sentiment_score', 0)
                nps.at[idx, 'is_contradictory'] = result.get('is_contradictory', False)
                nps.at[idx, 'contradiction_note'] = result.get('explanation', '')

                if result.get('is_contradictory'):
                    print(f"    ⚠️  CONTRADICTION — Account {nps.at[idx, 'account_id']}: "
                          f"Score={score}, Comment='{comment[:50]}...', "
                          f"Reason: {result.get('explanation', '')[:80]}")
            except Exception as e:
                print(f"    WARNING: Contradiction check failed for account {nps.at[idx, 'account_id']}: {e}")

    # Also flag the obvious ones: score mismatches with generic positive comments
    generic_positive_mask = (
            nps['is_generic_template'] &
            (nps['score'] <= 4) &
            (nps['verbatim_comment'] != '')
    )
    nps.loc[generic_positive_mask, 'is_contradictory'] = True
    nps.loc[generic_positive_mask, 'contradiction_note'] = (
        'Low NPS score with generic positive template comment — likely survey fatigue or copied text'
    )

    for idx in nps[generic_positive_mask].index:
        print(f"    ⚠️  GENERIC CONTRADICTION — Account {nps.at[idx, 'account_id']}: "
              f"Score={nps.at[idx, 'score']}, Comment='{nps.at[idx, 'verbatim_comment'][:50]}...'")

    return nps


# ============================================================
# 4. SDK / CHANGELOG RISK CROSS-REFERENCE
# ============================================================

def flag_sdk_risks(
        usage_df: pd.DataFrame,
        accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Cross-reference SDK versions from usage data with changelog deprecations.
    Returns a DataFrame with one row per account and their SDK risk status.
    """
    # Get the LATEST sdk_version per account (most recent month)
    latest_usage = usage_df.sort_values('month').groupby('account_id').last().reset_index()
    sdk_info = latest_usage[['account_id', 'sdk_version']].copy()

    def assess_sdk_risk(version_str: str) -> Dict[str, Any]:
        """Assess risk based on SDK version."""
        if not version_str or version_str == 'nan':
            return {
                'sdk_risk_level': 'unknown',
                'sdk_risk_details': 'No SDK version data available'
            }

        # Parse version
        version_str = version_str.strip().lower()

        # Extract major.minor version
        match = re.match(r'v?(\d+)\.(\d+)', version_str)
        if not match:
            return {
                'sdk_risk_level': 'unknown',
                'sdk_risk_details': f'Cannot parse version: {version_str}'
            }

        major = int(match.group(1))
        minor = int(match.group(2))

        risks = []
        risk_level = 'low'

        # v3.x — CRITICAL: sunset April 30, 2026, no security patches after
        if major <= 3:
            risk_level = 'critical'
            risks.append(
                f'SDK {version_str} is v3.x — REST API v2 endpoints sunset April 30, 2026. '
                f'Security patches end same date. Must migrate to v4.x immediately.'
            )

        # v4.0.x or v4.1.x — locale fallback bug (fixed in v4.2.3)
        elif major == 4 and minor < 2:
            risk_level = 'high'
            risks.append(
                f'SDK {version_str} has known locale fallback bug (fixed in v4.2.3). '
                f'Also missing breaking change: response.entry → response.data in v4.2.0+.'
            )

        # v4.2.x — missing latest improvements but functional
        elif major == 4 and minor == 2:
            risk_level = 'low'
            risks.append(f'SDK {version_str} is current but missing v4.3.x improvements (Agent OS, TypeScript types).')

        # v4.3.x — latest, all good
        elif major == 4 and minor >= 3:
            risk_level = 'none'
            risks.append(f'SDK {version_str} is latest. No version risks.')

        return {
            'sdk_risk_level': risk_level,
            'sdk_risk_details': ' | '.join(risks) if risks else 'No risks identified'
        }

    # Apply SDK risk assessment
    sdk_assessments = sdk_info['sdk_version'].apply(assess_sdk_risk)
    sdk_info['sdk_risk_level'] = sdk_assessments.apply(lambda x: x['sdk_risk_level'])
    sdk_info['sdk_risk_details'] = sdk_assessments.apply(lambda x: x['sdk_risk_details'])

    print(f"[Reconciliation] SDK risk assessment:")
    print(f"  - Critical (v3.x): {(sdk_info['sdk_risk_level'] == 'critical').sum()} accounts")
    print(f"  - High (v4.0-v4.1): {(sdk_info['sdk_risk_level'] == 'high').sum()} accounts")
    print(f"  - Low (v4.2): {(sdk_info['sdk_risk_level'] == 'low').sum()} accounts")
    print(f"  - None (v4.3+): {(sdk_info['sdk_risk_level'] == 'none').sum()} accounts")

    return sdk_info


# ============================================================
# 5. CROSS-SOURCE CONFLICT DETECTION
# ============================================================

def detect_cross_source_conflicts(
        accounts_df: pd.DataFrame,
        nps_processed: pd.DataFrame,
        csm_notes_matched: List[Dict],
        support_df: pd.DataFrame,
        usage_df: pd.DataFrame
) -> List[Dict[str, Any]]:
    """
    Identify conflicting signals across data sources for each account.
    These conflicts are themselves important risk signals.
    """
    conflicts = []

    for _, account in accounts_df.iterrows():
        aid = account['account_id']
        account_conflicts = []

        # Get NPS for this account
        nps_row = nps_processed[nps_processed['account_id'] == aid]
        nps_score = nps_row['score'].values[0] if len(nps_row) > 0 else None
        nps_contradictory = nps_row['is_contradictory'].values[0] if len(nps_row) > 0 else False

        # Get CSM notes for this account
        account_notes = [n for n in csm_notes_matched if n.get('matched_account_id') == aid]
        csm_sentiment = None
        if account_notes:
            sentiments = [n.get('sentiment', 'unknown') for n in account_notes]
            # Prioritize negative sentiment
            if 'negative' in sentiments:
                csm_sentiment = 'negative'
            elif 'mixed' in sentiments:
                csm_sentiment = 'mixed'
            elif 'positive' in sentiments:
                csm_sentiment = 'positive'
            else:
                csm_sentiment = 'neutral'

        # Conflict 1: NPS score contradicts comment
        if nps_contradictory:
            account_conflicts.append({
                'type': 'nps_score_comment_mismatch',
                'severity': 'high',
                'detail': f"NPS score ({nps_score}) contradicts the comment sentiment"
            })

        # Conflict 2: High NPS but negative CSM notes
        if nps_score and nps_score >= 8 and csm_sentiment in ['negative', 'mixed']:
            account_conflicts.append({
                'type': 'nps_vs_csm_sentiment',
                'severity': 'high',
                'detail': f"NPS score is {nps_score} (positive) but CSM notes indicate {csm_sentiment} sentiment"
            })

        # Conflict 3: Low NPS but positive CSM notes
        if nps_score and nps_score <= 5 and csm_sentiment == 'positive':
            account_conflicts.append({
                'type': 'nps_vs_csm_sentiment',
                'severity': 'medium',
                'detail': f"NPS score is {nps_score} (low) but CSM notes indicate positive sentiment"
            })

        # Conflict 4: CSM notes mention "all good" but support tickets are heavy
        account_tickets = support_df[support_df['account_id'] == aid]
        recent_p1_tickets = account_tickets[
            (account_tickets['priority'] == 'P1') &
            (account_tickets['status'].isin(['Open', 'Escalated']))
            ]

        if csm_sentiment == 'positive' and len(recent_p1_tickets) >= 2:
            account_conflicts.append({
                'type': 'csm_vs_support_tickets',
                'severity': 'medium',
                'detail': f"CSM notes are positive but account has {len(recent_p1_tickets)} open/escalated P1 tickets"
            })

        # Conflict 5: Generic NPS comment with extreme score
        if len(nps_row) > 0:
            is_generic = nps_row['is_generic_template'].values[0]
            if is_generic and (nps_score <= 3 or nps_score >= 9):
                account_conflicts.append({
                    'type': 'generic_nps_extreme_score',
                    'severity': 'medium',
                    'detail': f"NPS score {nps_score} with generic/templated comment — signal may be unreliable"
                })

        if account_conflicts:
            conflicts.append({
                'account_id': aid,
                'account_name': account['account_name'],
                'conflicts': account_conflicts,
                'conflict_count': len(account_conflicts)
            })

    print(f"[Reconciliation] Found conflicts in {len(conflicts)} accounts:")
    for c in conflicts:
        print(f"  - {c['account_name']} ({c['account_id']}): {c['conflict_count']} conflicts")

    return conflicts


# ============================================================
# 6. MASTER RECONCILIATION PIPELINE
# ============================================================

def reconcile_all_data(
        accounts_df: pd.DataFrame,
        usage_df: pd.DataFrame,
        support_df: pd.DataFrame,
        csm_notes_raw: str,
        nps_df: pd.DataFrame,
        changelog_raw: str,
        llm_provider: str = "groq"
) -> Dict[str, Any]:
    """
    Master reconciliation function. Runs all steps and returns enriched data.
    """
    print("\n" + "=" * 60)
    print("DATA RECONCILIATION")
    print("=" * 60)

    # Step 1: Filter to renewal window
    renewal_accounts = accounts_df[
        (accounts_df['contract_end_date'].dt.date >= REFERENCE_DATE) &
        (accounts_df['contract_end_date'].dt.date <= RENEWAL_CUTOFF_DATE)
        ].copy()
    print(f"\n[Step 1] Accounts in renewal window ({REFERENCE_DATE} to {RENEWAL_CUTOFF_DATE}): {len(renewal_accounts)}")

    # Step 2: Parse CSM notes
    print(f"\n[Step 2] Parsing CSM notes with LLM...")
    note_entries = split_csm_notes(csm_notes_raw)
    parsed_notes = parse_csm_notes_with_llm(note_entries, provider=llm_provider)

    # Step 3: Match CSM notes to accounts
    print(f"\n[Step 3] Matching CSM notes to accounts...")
    account_lookup = build_account_lookup(accounts_df)
    matched_notes = []
    for note in parsed_notes:
        matched = match_note_to_account(note, account_lookup)
        matched_notes.append(matched)

    # Report matching stats
    matched_count = sum(1 for n in matched_notes if n.get('matched_account_id') is not None)
    unmatched = [n for n in matched_notes if n.get('matched_account_id') is None]
    print(f"  Matched: {matched_count}/{len(matched_notes)}")
    if unmatched:
        print(f"  Unmatched notes: {[n.get('account_name_raw', '?') for n in unmatched]}")

    # Step 4: Process NPS
    print(f"\n[Step 4] Processing NPS responses (translate + contradiction detection)...")
    nps_processed = translate_and_analyze_nps(nps_df, provider=llm_provider)

    # Step 5: SDK risk assessment
    print(f"\n[Step 5] Assessing SDK version risks...")
    sdk_risks = flag_sdk_risks(usage_df, accounts_df)

    # Step 6: Cross-source conflict detection
    print(f"\n[Step 6] Detecting cross-source conflicts...")
    conflicts = detect_cross_source_conflicts(
        accounts_df, nps_processed, matched_notes, support_df, usage_df
    )

    # Package results
    reconciled = {
        'all_accounts': accounts_df,
        'renewal_accounts': renewal_accounts,
        'usage_metrics': usage_df,
        'support_tickets': support_df,
        'csm_notes_parsed': matched_notes,
        'nps_processed': nps_processed,
        'sdk_risks': sdk_risks,
        'conflicts': conflicts,
        'changelog_raw': changelog_raw
    }

    print("\n" + "=" * 60)
    print("RECONCILIATION COMPLETE")
    print(f"  Renewal accounts: {len(renewal_accounts)}")
    print(f"  CSM notes parsed & matched: {matched_count}")
    print(f"  NPS responses processed: {len(nps_processed)}")
    print(f"  SDK risks flagged: {(sdk_risks['sdk_risk_level'].isin(['critical', 'high'])).sum()}")
    print(f"  Cross-source conflicts: {len(conflicts)}")
    print("=" * 60)

    return reconciled