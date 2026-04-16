"""
Data Ingestion Module
Loads all raw data files and performs initial type casting / validation.
No business logic here — just clean loading.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Any
from src.config import (
    ACCOUNTS_PATH, USAGE_METRICS_PATH, SUPPORT_TICKETS_PATH,
    CSM_NOTES_PATH, NPS_RESPONSES_PATH, CHANGELOG_PATH
)


def load_accounts(path: Path = ACCOUNTS_PATH) -> pd.DataFrame:
    """Load accounts.csv with proper dtypes."""
    df = pd.read_csv(path)

    # Ensure correct types
    df['account_id'] = df['account_id'].astype(int)
    df['arr'] = df['arr'].astype(float)
    df['contract_end_date'] = pd.to_datetime(df['contract_end_date'])

    # Strip whitespace from string columns
    str_cols = ['account_name', 'plan_tier', 'industry', 'csm_name', 'region']
    for col in str_cols:
        df[col] = df[col].str.strip()

    print(f"[Ingestion] Loaded {len(df)} accounts")
    print(f"  - Plan tiers: {df['plan_tier'].value_counts().to_dict()}")
    print(f"  - Regions: {df['region'].value_counts().to_dict()}")
    print(f"  - Contract end range: {df['contract_end_date'].min().date()} to {df['contract_end_date'].max().date()}")

    return df


def load_usage_metrics(path: Path = USAGE_METRICS_PATH) -> pd.DataFrame:
    """Load usage_metrics.csv with proper dtypes."""
    df = pd.read_csv(path)

    df['account_id'] = df['account_id'].astype(int)
    df['month'] = pd.to_datetime(df['month'] + '-01')  # Convert YYYY-MM to date
    df['api_calls'] = df['api_calls'].astype(int)
    df['content_entries_created'] = df['content_entries_created'].astype(int)
    df['active_users'] = df['active_users'].astype(int)
    df['workflows_triggered'] = df['workflows_triggered'].astype(int)
    df['sdk_version'] = df['sdk_version'].astype(str).str.strip()

    print(f"[Ingestion] Loaded {len(df)} usage metric rows")
    print(f"  - Unique accounts: {df['account_id'].nunique()}")
    print(f"  - Date range: {df['month'].min().date()} to {df['month'].max().date()}")
    print(f"  - SDK versions: {df['sdk_version'].unique().tolist()}")

    return df


def load_support_tickets(path: Path = SUPPORT_TICKETS_PATH) -> pd.DataFrame:
    """Load support_tickets.csv with proper dtypes."""
    df = pd.read_csv(path)

    df['account_id'] = df['account_id'].astype(int)

    # Parse dates - handle potential format variations
    date_cols = ['created_date', 'resolved_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Strip string columns
    str_cols = [c for c in df.columns if df[c].dtype == 'object']
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    print(f"[Ingestion] Loaded {len(df)} support tickets")
    print(f"  - Unique accounts: {df['account_id'].nunique()}")
    print(f"  - Columns: {df.columns.tolist()}")
    if 'priority' in df.columns:
        print(f"  - Priority distribution: {df['priority'].value_counts().to_dict()}")
    if 'status' in df.columns:
        print(f"  - Status distribution: {df['status'].value_counts().to_dict()}")

    return df


def load_csm_notes(path: Path = CSM_NOTES_PATH) -> str:
    """Load csm_notes.txt as raw text."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"[Ingestion] Loaded CSM notes: {len(text)} characters")

    # Count approximate number of note entries (separated by ---)
    note_count = text.count('---') + 1
    print(f"  - Approximate note entries: {note_count}")

    return text


def load_nps_responses(path: Path = NPS_RESPONSES_PATH) -> pd.DataFrame:
    """Load nps_responses.csv."""
    df = pd.read_csv(path)

    df['account_id'] = df['account_id'].astype(int)
    df['score'] = df['score'].astype(int)

    # Keep verbatim_comment as string, handle NaN
    df['verbatim_comment'] = df['verbatim_comment'].fillna('').astype(str).str.strip()

    print(f"[Ingestion] Loaded {len(df)} NPS responses")
    print(f"  - Score range: {df['score'].min()} to {df['score'].max()}")
    print(f"  - Mean score: {df['score'].mean():.1f}")
    print(f"  - Responses with comments: {(df['verbatim_comment'] != '').sum()}")

    # Flag non-English comments
    non_english = df[df['verbatim_comment'].str.contains(
        r'[^\x00-\x7F]', regex=True, na=False
    )]
    if len(non_english) > 0:
        print(f"  - Non-English comments found: {non_english['account_id'].tolist()}")

    return df


def load_changelog(path: Path = CHANGELOG_PATH) -> str:
    """Load changelog.md as raw text."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"[Ingestion] Loaded changelog: {len(text)} characters")

    return text


def load_all_data() -> Dict[str, Any]:
    """
    Load all data sources and return as a dictionary.
    This is the main entry point for data ingestion.
    """
    print("=" * 60)
    print("DATA INGESTION")
    print("=" * 60)

    data = {
        'accounts': load_accounts(),
        'usage_metrics': load_usage_metrics(),
        'support_tickets': load_support_tickets(),
        'csm_notes_raw': load_csm_notes(),
        'nps_responses': load_nps_responses(),
        'changelog_raw': load_changelog()
    }

    print("=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)

    return data


# --- Quick validation when run directly ---
if __name__ == "__main__":
    data = load_all_data()

    # Cross-reference check
    account_ids = set(data['accounts']['account_id'])
    usage_ids = set(data['usage_metrics']['account_id'])
    nps_ids = set(data['nps_responses']['account_id'])

    print(f"\n--- Cross-reference Check ---")
    print(f"Accounts: {len(account_ids)}")
    print(f"Accounts with usage data: {len(account_ids & usage_ids)}")
    print(f"Accounts missing usage data: {account_ids - usage_ids}")
    print(f"Accounts with NPS: {len(account_ids & nps_ids)}")
    print(f"Accounts missing NPS: {len(account_ids - nps_ids)}")
    print(f"NPS IDs not in accounts: {nps_ids - account_ids}")