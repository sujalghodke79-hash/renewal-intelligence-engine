"""
Test Part 2 — Data Reconciliation.
Run: python test_part2.py
"""

import sys

sys.path.insert(0, '..')

from src.data_ingestion import load_all_data
from src.data_reconciliation import (
    split_csm_notes,
    build_account_lookup,
    match_note_to_account,
    reconcile_all_data
)


def test_csm_splitting():
    """Test that CSM notes split correctly."""
    from src.data_ingestion import load_csm_notes
    raw = load_csm_notes()
    notes = split_csm_notes(raw)

    print(f"\n--- CSM Note Splitting ---")
    print(f"Total notes: {len(notes)}")
    for i, note in enumerate(notes):
        print(f"\n  Note {i + 1} (first 80 chars): {note[:80]}...")

    assert len(notes) >= 20, f"Expected at least 20 notes, got {len(notes)}"
    print("\n✅ CSM splitting test passed!")


def test_fuzzy_matching():
    """Test fuzzy matching with known typos from the data."""
    from src.data_ingestion import load_accounts
    accounts = load_accounts()
    lookup = build_account_lookup(accounts)

    test_cases = [
        {"account_name_raw": "BritePath Solutions", "expected": "BrightPath Solutions"},
        {"account_name_raw": "Pinacle Media", "expected": "Pinnacle Media Group"},
        {"account_name_raw": "Thunderbolt Moters", "expected": "Thunderbolt Motors"},
        {"account_name_raw": "Acme Corp", "expected": "Acme Corp"},
        {"account_name_raw": "vanguard retail", "expected": "Vanguard Retail"},
    ]

    print(f"\n--- Fuzzy Matching Tests ---")
    all_passed = True
    for tc in test_cases:
        note = {"account_name_raw": tc["account_name_raw"], "account_id_if_mentioned": None}
        result = match_note_to_account(note, lookup)
        matched = result.get('matched_account_name', 'NO MATCH')
        status = "✅" if matched == tc["expected"] else "❌"
        if matched != tc["expected"]:
            all_passed = False
        print(f"  {status} '{tc['account_name_raw']}' → '{matched}' (expected: '{tc['expected']}')")

    if all_passed:
        print("\n✅ All fuzzy matching tests passed!")
    else:
        print("\n⚠️  Some fuzzy matching tests failed — review results above")


def test_full_reconciliation():
    """Run full reconciliation pipeline."""
    data = load_all_data()

    reconciled = reconcile_all_data(
        accounts_df=data['accounts'],
        usage_df=data['usage_metrics'],
        support_df=data['support_tickets'],
        csm_notes_raw=data['csm_notes_raw'],
        nps_df=data['nps_responses'],
        changelog_raw=data['changelog_raw'],
        llm_provider="groq"  # Change to "openrouter" if needed
    )

    # Validate outputs
    assert 'renewal_accounts' in reconciled
    assert 'csm_notes_parsed' in reconciled
    assert 'nps_processed' in reconciled
    assert 'sdk_risks' in reconciled
    assert 'conflicts' in reconciled

    print(f"\n--- Reconciliation Results Summary ---")
    print(f"Renewal accounts: {len(reconciled['renewal_accounts'])}")

    # Show parsed CSM notes
    print(f"\nParsed CSM Notes:")
    for note in reconciled['csm_notes_parsed']:
        aid = note.get('matched_account_id', '?')
        name = note.get('matched_account_name', note.get('account_name_raw', '?'))
        sentiment = note.get('sentiment', '?')
        risks = note.get('churn_risk_signals', [])
        competitors = note.get('competitors_mentioned', [])
        print(f"  [{aid}] {name} — sentiment={sentiment}, risks={len(risks)}, competitors={competitors}")

    # Show NPS contradictions
    nps = reconciled['nps_processed']
    contradictions = nps[nps['is_contradictory'] == True]
    print(f"\nNPS Contradictions ({len(contradictions)}):")
    for _, row in contradictions.iterrows():
        print(f"  Account {row['account_id']}: score={row['score']}, "
              f"comment='{row['translated_comment'][:60]}...', "
              f"reason='{row['contradiction_note'][:80]}'")

    # Show SDK risks
    sdk = reconciled['sdk_risks']
    critical = sdk[sdk['sdk_risk_level'].isin(['critical', 'high'])]
    print(f"\nSDK Risks ({len(critical)} critical/high):")
    for _, row in critical.iterrows():
        print(f"  Account {row['account_id']}: {row['sdk_version']} — {row['sdk_risk_level']}")

    print("\n✅ Full reconciliation test passed!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Run only non-LLM tests")
    args = parser.parse_args()

    test_csm_splitting()
    test_fuzzy_matching()

    if not args.quick:
        print("\n\n🚀 Running full reconciliation (requires LLM API)...")
        test_full_reconciliation()
    else:
        print("\n⏭️  Skipping LLM tests (use without --quick to run full pipeline)")