"""
Quick test to validate Part 1 — Data Ingestion.
Run from project root: python test_part1.py
"""

import sys

sys.path.insert(0, '.')

from src.data_ingestion import load_all_data


def test_ingestion():
    data = load_all_data()

    # Basic assertions
    assert len(data['accounts']) == 120, f"Expected 120 accounts, got {len(data['accounts'])}"
    assert 'account_id' in data['accounts'].columns
    assert 'contract_end_date' in data['accounts'].columns

    # Usage metrics should have 6 months per account
    usage = data['usage_metrics']
    months_per_account = usage.groupby('account_id')['month'].count()
    print(f"\nMonths per account — min: {months_per_account.min()}, max: {months_per_account.max()}")

    # NPS should be subset of accounts
    nps_ids = set(data['nps_responses']['account_id'])
    account_ids = set(data['accounts']['account_id'])
    orphan_nps = nps_ids - account_ids
    if orphan_nps:
        print(f"WARNING: NPS responses for non-existent accounts: {orphan_nps}")

    # CSM notes should be non-empty
    assert len(data['csm_notes_raw']) > 100, "CSM notes seem too short"

    # Changelog should be non-empty
    assert len(data['changelog_raw']) > 100, "Changelog seems too short"

    print("\n✅ All Part 1 tests passed!")

    # Print some useful stats for next steps
    accounts = data['accounts']
    from src.config import REFERENCE_DATE, RENEWAL_WINDOW_DAYS
    from datetime import timedelta

    cutoff = REFERENCE_DATE + timedelta(days=RENEWAL_WINDOW_DAYS)
    renewals = accounts[
        (accounts['contract_end_date'].dt.date >= REFERENCE_DATE) &
        (accounts['contract_end_date'].dt.date <= cutoff)
        ]
    print(f"\nAccounts renewing between {REFERENCE_DATE} and {cutoff}: {len(renewals)}")
    print(renewals[['account_id', 'account_name', 'arr', 'contract_end_date', 'plan_tier']].to_string(index=False))


if __name__ == "__main__":
    test_ingestion()