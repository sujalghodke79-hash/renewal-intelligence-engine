"""
Feature Engineering Module

Computes risk signals from each data source and builds the master feature matrix.
All CSV outputs use utf-8-sig encoding for Windows Excel compatibility.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import timedelta

from src.config import REFERENCE_DATE, RENEWAL_CUTOFF_DATE


# ============================================================
# 1. USAGE TREND FEATURES
# ============================================================

def compute_usage_features(usage_df: pd.DataFrame) -> pd.DataFrame:
    """Compute usage trend features per account."""
    features = []

    all_months = sorted(usage_df['month'].unique())

    if len(all_months) < 2:
        print("[Features] WARNING: Not enough months of usage data")
        return pd.DataFrame()

    midpoint = len(all_months) // 2
    older_months = all_months[:midpoint]
    recent_months = all_months[midpoint:]

    print(f"[Features] Usage periods:")
    print(f"  Older:  {[m.strftime('%Y-%m') for m in older_months]}")
    print(f"  Recent: {[m.strftime('%Y-%m') for m in recent_months]}")

    metrics = [
        'api_calls', 'content_entries_created',
        'active_users', 'workflows_triggered'
    ]

    for account_id, group in usage_df.groupby('account_id'):
        row = {'account_id': account_id}

        older_data = group[group['month'].isin(older_months)]
        recent_data = group[group['month'].isin(recent_months)]

        for metric in metrics:
            older_vals = (
                older_data[metric].values
                if len(older_data) > 0
                else np.array([0])
            )
            recent_vals = (
                recent_data[metric].values
                if len(recent_data) > 0
                else np.array([0])
            )

            older_avg = float(np.mean(older_vals))
            recent_avg = float(np.mean(recent_vals))

            if older_avg > 0:
                pct_change = ((recent_avg - older_avg) / older_avg) * 100
            elif recent_avg > 0:
                pct_change = 100.0
            else:
                pct_change = 0.0

            all_vals = group[metric].values

            row[f'{metric}_older_avg'] = round(older_avg, 2)
            row[f'{metric}_recent_avg'] = round(recent_avg, 2)
            row[f'{metric}_pct_change'] = round(pct_change, 2)
            row[f'{metric}_total'] = int(np.sum(all_vals))
            row[f'{metric}_std'] = round(float(np.std(all_vals)), 2)
            row[f'{metric}_min'] = int(np.min(all_vals))
            row[f'{metric}_max'] = int(np.max(all_vals))
            row[f'{metric}_last'] = int(all_vals[-1]) if len(all_vals) > 0 else 0

            if len(recent_vals) >= 2:
                x = np.arange(len(recent_vals))
                slope = float(np.polyfit(x, recent_vals, 1)[0])
                row[f'{metric}_recent_slope'] = round(slope, 2)
            else:
                row[f'{metric}_recent_slope'] = 0.0

        row['api_calls_declining'] = row['api_calls_pct_change'] < -20
        row['content_creation_declining'] = row['content_entries_created_pct_change'] < -20
        row['active_users_declining'] = row['active_users_pct_change'] < -20
        row['workflows_declining'] = row['workflows_triggered_pct_change'] < -30
        row['api_calls_severe_decline'] = row['api_calls_pct_change'] < -50
        row['active_users_severe_decline'] = row['active_users_pct_change'] < -50
        row['zero_api_calls_recent'] = row['api_calls_recent_avg'] == 0
        row['zero_active_users_recent'] = row['active_users_recent_avg'] == 0
        row['zero_workflows_recent'] = row['workflows_triggered_recent_avg'] == 0
        row['workflow_abandoned'] = (
            row['workflows_triggered_older_avg'] > 1 and
            row['workflows_triggered_recent_avg'] <= 0.5
        )

        decline_signals = [
            row['api_calls_declining'], row['content_creation_declining'],
            row['active_users_declining'], row['workflows_declining'],
            row['api_calls_severe_decline'], row['active_users_severe_decline'],
            row['workflow_abandoned']
        ]
        decline_count = sum(1 for s in decline_signals if s)
        row['usage_health_score'] = max(0, 100 - decline_count * 15)
        row['declining_metrics_count'] = sum(1 for s in [
            row['api_calls_declining'], row['content_creation_declining'],
            row['active_users_declining'], row['workflows_declining']
        ] if s)

        # Get the latest SDK version for this account
        latest = group.sort_values('month').iloc[-1]
        row['sdk_version'] = str(latest.get('sdk_version', 'unknown'))

        features.append(row)

    df = pd.DataFrame(features)

    print(f"[Features] Usage features computed for {len(df)} accounts")
    print(f"  Declining API calls: {df['api_calls_declining'].sum()}")
    print(f"  Declining active users: {df['active_users_declining'].sum()}")
    print(f"  Workflow abandoned: {df['workflow_abandoned'].sum()}")

    return df


# ============================================================
# 2. SUPPORT TICKET FEATURES
# ============================================================

def compute_support_features(
    support_df: pd.DataFrame,
    accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute support ticket features per account."""
    recent_cutoff = pd.Timestamp(REFERENCE_DATE - timedelta(days=90))
    features = []
    all_account_ids = accounts_df['account_id'].unique()

    for account_id in all_account_ids:
        row = {'account_id': account_id}
        account_tickets = support_df[support_df['account_id'] == account_id]

        if len(account_tickets) == 0:
            row.update({
                'total_tickets': 0, 'p1_tickets': 0, 'p2_tickets': 0,
                'p3_tickets': 0, 'p4_tickets': 0, 'open_tickets': 0,
                'escalated_tickets': 0, 'open_p1_tickets': 0,
                'open_escalated_critical': 0, 'avg_resolution_hours': 0,
                'max_resolution_hours': 0, 'recent_ticket_count': 0,
                'recent_p1_count': 0, 'ticket_velocity_trend': 0,
                'has_deprecation_tickets': False, 'has_migration_tickets': False,
                'has_blocking_tickets': False, 'has_recurring_tickets': False,
                'support_health_score': 80, 'no_tickets_flag': True
            })
            features.append(row)
            continue

        row['total_tickets'] = len(account_tickets)
        row['p1_tickets'] = len(account_tickets[account_tickets['priority'] == 'P1'])
        row['p2_tickets'] = len(account_tickets[account_tickets['priority'] == 'P2'])
        row['p3_tickets'] = len(account_tickets[account_tickets['priority'] == 'P3'])
        row['p4_tickets'] = len(account_tickets[account_tickets['priority'] == 'P4'])
        row['open_tickets'] = len(account_tickets[account_tickets['status'] == 'Open'])
        row['escalated_tickets'] = len(account_tickets[account_tickets['status'] == 'Escalated'])
        row['open_p1_tickets'] = len(account_tickets[
            (account_tickets['priority'] == 'P1') &
            (account_tickets['status'].isin(['Open', 'Escalated']))
        ])
        row['open_escalated_critical'] = len(account_tickets[
            (account_tickets['priority'].isin(['P1', 'P2'])) &
            (account_tickets['status'].isin(['Open', 'Escalated']))
        ])

        resolved = account_tickets[account_tickets['resolution_time_hours'].notna()]
        if len(resolved) > 0:
            row['avg_resolution_hours'] = round(
                float(resolved['resolution_time_hours'].mean()), 1
            )
            row['max_resolution_hours'] = round(
                float(resolved['resolution_time_hours'].max()), 1
            )
        else:
            row['avg_resolution_hours'] = 0
            row['max_resolution_hours'] = 0

        recent = account_tickets[account_tickets['created_date'] >= recent_cutoff]
        older = account_tickets[account_tickets['created_date'] < recent_cutoff]
        row['recent_ticket_count'] = len(recent)
        row['recent_p1_count'] = len(recent[recent['priority'] == 'P1'])

        if len(older) > 0:
            row['ticket_velocity_trend'] = round(len(recent) / 3 - len(older) / 3, 2)
        else:
            row['ticket_velocity_trend'] = 0

        subjects = ' '.join(account_tickets['subject'].str.lower().tolist())
        descriptions = ' '.join(account_tickets['description'].str.lower().tolist())
        all_text = subjects + ' ' + descriptions

        row['has_deprecation_tickets'] = any(t in all_text for t in [
            'deprecat', 'sunset', 'rest api', 'sdk upgrade', 'migration'
        ])
        row['has_migration_tickets'] = any(t in all_text for t in [
            'migration', 'legacy editor', 'sdk upgrade'
        ])
        row['has_blocking_tickets'] = 'blocking' in all_text or 'impacting go-live' in all_text
        row['has_recurring_tickets'] = 'recurring' in all_text or 'third time' in all_text

        score = 100
        score -= min(row['open_p1_tickets'] * 20, 40)
        score -= min(row['escalated_tickets'] * 10, 20)
        score -= min(row['p1_tickets'] * 5, 25)
        score -= 15 if row['has_blocking_tickets'] else 0
        score -= 10 if row['has_recurring_tickets'] else 0
        score -= 10 if row['has_deprecation_tickets'] else 0
        score -= 5 if row['avg_resolution_hours'] > 72 else 0
        row['support_health_score'] = max(0, score)
        row['no_tickets_flag'] = False

        features.append(row)

    df = pd.DataFrame(features)

    print(f"[Features] Support features computed for {len(df)} accounts")
    print(f"  Open P1 tickets: {(df['open_p1_tickets'] > 0).sum()}")
    print(f"  Escalated tickets: {(df['escalated_tickets'] > 0).sum()}")
    print(f"  Deprecation tickets: {df['has_deprecation_tickets'].sum()}")

    return df


# ============================================================
# 3. NPS FEATURES
# ============================================================

def compute_nps_features(
    nps_processed: pd.DataFrame,
    accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute NPS-derived features per account."""
    features = []
    all_account_ids = accounts_df['account_id'].unique()

    for account_id in all_account_ids:
        row = {'account_id': account_id}
        nps_row = nps_processed[nps_processed['account_id'] == account_id]

        if len(nps_row) == 0:
            row.update({
                'nps_score': None, 'nps_category': 'no_response',
                'nps_has_response': False, 'nps_has_comment': False,
                'nps_is_detractor': False, 'nps_is_passive': False,
                'nps_is_promoter': False, 'nps_is_contradictory': False,
                'nps_is_generic': False, 'nps_is_non_english': False,
                'nps_comment_sentiment': None, 'nps_health_score': 50
            })
            features.append(row)
            continue

        nps_data = nps_row.iloc[0]
        score = int(nps_data['score'])

        row['nps_score'] = score
        row['nps_has_response'] = True
        row['nps_has_comment'] = bool(
            nps_data['verbatim_comment'] and
            str(nps_data['verbatim_comment']).strip()
        )

        if score <= 6:
            row['nps_category'] = 'detractor'
            row['nps_is_detractor'] = True
            row['nps_is_passive'] = False
            row['nps_is_promoter'] = False
        elif score <= 8:
            row['nps_category'] = 'passive'
            row['nps_is_detractor'] = False
            row['nps_is_passive'] = True
            row['nps_is_promoter'] = False
        else:
            row['nps_category'] = 'promoter'
            row['nps_is_detractor'] = False
            row['nps_is_passive'] = False
            row['nps_is_promoter'] = True

        row['nps_is_contradictory'] = bool(nps_data.get('is_contradictory', False))
        row['nps_is_generic'] = bool(nps_data.get('is_generic_template', False))
        row['nps_is_non_english'] = nps_data.get('detected_language', 'en') != 'en'
        row['nps_comment_sentiment'] = nps_data.get('comment_sentiment_score', None)

        if row['nps_is_contradictory']:
            nps_health = 35 if score >= 7 else 30
        elif row['nps_is_generic']:
            nps_health = 40 if score <= 4 else 60
        else:
            nps_health = score * 10

        row['nps_health_score'] = nps_health
        features.append(row)

    df = pd.DataFrame(features)

    print(f"[Features] NPS features computed for {len(df)} accounts")
    print(f"  Detractors: {df['nps_is_detractor'].sum()}")
    print(f"  Passives: {df['nps_is_passive'].sum()}")
    print(f"  Promoters: {df['nps_is_promoter'].sum()}")
    print(f"  Contradictory: {df['nps_is_contradictory'].sum()}")

    return df


# ============================================================
# 4. CSM NOTES FEATURES
# ============================================================

def compute_csm_features(
    csm_notes_parsed: List[Dict],
    accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute features from parsed CSM notes per account."""
    features = []
    all_account_ids = accounts_df['account_id'].unique()

    for account_id in all_account_ids:
        row = {'account_id': account_id}
        account_notes = [
            n for n in csm_notes_parsed
            if n.get('matched_account_id') == account_id
        ]

        if len(account_notes) == 0:
            row.update({
                'csm_has_notes': False, 'csm_note_count': 0,
                'csm_sentiment': 'unknown', 'csm_churn_signal_count': 0,
                'csm_expansion_signal_count': 0, 'csm_competitor_count': 0,
                'csm_competitors_list': '', 'csm_has_executive_involvement': False,
                'csm_has_missed_meetings': False, 'csm_has_budget_concerns': False,
                'csm_has_evaluation_signals': False, 'csm_has_compliance_blockers': False,
                'csm_has_migration_issues': False, 'csm_action_items_count': 0,
                'csm_health_score': 50
            })
            features.append(row)
            continue

        row['csm_has_notes'] = True
        row['csm_note_count'] = len(account_notes)

        all_churn_signals = []
        all_expansion_signals = []
        all_competitors = []
        all_stakeholders = []
        all_action_items = []
        sentiments = []

        for note in account_notes:
            all_churn_signals.extend(note.get('churn_risk_signals', []))
            all_expansion_signals.extend(note.get('expansion_signals', []))
            all_competitors.extend(note.get('competitors_mentioned', []))
            all_stakeholders.extend(note.get('key_stakeholders', []))
            all_action_items.extend(note.get('action_items', []))
            sentiments.append(note.get('sentiment', 'neutral'))

        row['csm_churn_signal_count'] = len(all_churn_signals)
        row['csm_expansion_signal_count'] = len(all_expansion_signals)
        row['csm_competitor_count'] = len(set(all_competitors))
        row['csm_competitors_list'] = (
            ', '.join(set(all_competitors)) if all_competitors else ''
        )
        row['csm_action_items_count'] = len(all_action_items)

        if 'negative' in sentiments:
            row['csm_sentiment'] = 'negative'
        elif 'mixed' in sentiments:
            row['csm_sentiment'] = 'mixed'
        elif 'positive' in sentiments:
            row['csm_sentiment'] = 'positive'
        else:
            row['csm_sentiment'] = 'neutral'

        all_text = ' '.join([
            ' '.join(note.get('churn_risk_signals', [])) +
            ' ' + note.get('summary', '') +
            ' ' + ' '.join(note.get('action_items', []))
            for note in account_notes
        ]).lower()

        stakeholders_text = ' '.join(all_stakeholders).lower()
        row['csm_has_executive_involvement'] = any(
            t in stakeholders_text
            for t in ['vp', 'cto', 'cro', 'ciso', 'ceo', 'cfo', 'chief',
                       'vice president', 'director']
        )
        row['csm_has_missed_meetings'] = any(
            t in all_text
            for t in ['no show', 'no-show', 'missed qbr', 'missed meeting',
                       "haven't responded", "hasn't responded", 'not responded']
        )
        row['csm_has_budget_concerns'] = any(
            t in all_text
            for t in ['budget cut', 'budget', 'price increase', 'pricing',
                       'discount', 'cost', 'match', 'competing offer']
        )
        row['csm_has_evaluation_signals'] = any(
            t in all_text
            for t in ['evaluat', 'poc', 'explore options', 'competitor',
                       'alternative', 'hygraph', 'strapi', 'sanity', 'contentful',
                       'kontent', 'builder.io', 'wordpress']
        )
        row['csm_has_compliance_blockers'] = any(
            t in all_text
            for t in ['compliance', 'soc 2', 'security questionnaire', 'vendor security',
                       'audit', 'gdpr', 'regulated', 'single-tenancy', 'fips']
        )
        row['csm_has_migration_issues'] = any(
            t in all_text
            for t in ['migration', 'deprecat', 'sdk', 'v3', 'workaround',
                       'legacy', 'never finished', 'half their content']
        )

        score = 50
        if row['csm_sentiment'] == 'negative':
            score -= 25
        elif row['csm_sentiment'] == 'mixed':
            score -= 10

        score -= min(row['csm_churn_signal_count'] * 8, 30)
        score -= min(row['csm_competitor_count'] * 15, 30)
        score -= 15 if (
            row['csm_has_executive_involvement'] and
            row['csm_sentiment'] != 'positive'
        ) else 0
        score -= 10 if row['csm_has_missed_meetings'] else 0
        score -= 10 if row['csm_has_budget_concerns'] else 0
        score -= 15 if row['csm_has_evaluation_signals'] else 0
        score -= 10 if row['csm_has_compliance_blockers'] else 0
        score -= 5 if row['csm_has_migration_issues'] else 0

        if row['csm_sentiment'] == 'positive':
            score += 25
        score += min(row['csm_expansion_signal_count'] * 10, 25)

        row['csm_health_score'] = max(0, min(100, score))
        features.append(row)

    df = pd.DataFrame(features)

    print(f"[Features] CSM features computed for {len(df)} accounts")
    print(f"  With notes: {df['csm_has_notes'].sum()}")
    print(f"  Negative sentiment: {(df['csm_sentiment'] == 'negative').sum()}")
    print(f"  Competitor mentions: {(df['csm_competitor_count'] > 0).sum()}")

    return df


# ============================================================
# 5. SDK FEATURES
# ============================================================

def compute_sdk_features(sdk_risks: pd.DataFrame) -> pd.DataFrame:
    """Convert SDK risk data into numeric features."""
    df = sdk_risks.copy()

    risk_level_map = {
        'critical': 0, 'high': 20, 'low': 70, 'none': 100, 'unknown': 50
    }
    df['sdk_health_score'] = df['sdk_risk_level'].map(risk_level_map).fillna(50)
    df['sdk_is_critical'] = df['sdk_risk_level'] == 'critical'
    df['sdk_is_high_risk'] = df['sdk_risk_level'].isin(['critical', 'high'])
    df['sdk_is_v3'] = df['sdk_version'].str.startswith('v3', na=False)

    print(f"[Features] SDK features computed for {len(df)} accounts")
    print(f"  Critical: {df['sdk_is_critical'].sum()}")
    print(f"  High risk: {df['sdk_is_high_risk'].sum()}")

    return df[[
        'account_id', 'sdk_version', 'sdk_risk_level',
        'sdk_health_score', 'sdk_is_critical', 'sdk_is_high_risk', 'sdk_is_v3'
    ]]


# ============================================================
# 6. CONTRACT / FIRMOGRAPHIC FEATURES
# ============================================================

def compute_contract_features(accounts_df: pd.DataFrame) -> pd.DataFrame:
    """Compute contract and firmographic features."""
    df = accounts_df.copy()

    df['days_until_renewal'] = (
        df['contract_end_date'] - pd.Timestamp(REFERENCE_DATE)
    ).dt.days

    df['renewal_urgency'] = pd.cut(
        df['days_until_renewal'],
        bins=[-999, 15, 30, 60, 90, 9999],
        labels=['critical', 'urgent', 'approaching', 'upcoming', 'distant']
    )

    df['arr_tier'] = pd.cut(
        df['arr'],
        bins=[0, 50000, 200000, 500000, 1000000, float('inf')],
        labels=['small', 'mid_market', 'growth', 'large', 'enterprise']
    )

    plan_tier_map = {'Starter': 1, 'Growth': 2, 'Scale': 3, 'Enterprise': 4}
    df['plan_tier_numeric'] = df['plan_tier'].map(plan_tier_map)

    df['arr_plan_mismatch'] = (
        ((df['arr'] > 500000) & (df['plan_tier'].isin(['Starter', 'Growth']))) |
        ((df['arr'] < 30000) & (df['plan_tier'].isin(['Scale', 'Enterprise'])))
    )

    df['in_renewal_window'] = (
        (df['days_until_renewal'] >= 0) &
        (df['days_until_renewal'] <= 90)
    )

    columns_to_keep = [
        'account_id', 'account_name', 'arr', 'contract_end_date',
        'plan_tier', 'industry', 'csm_name', 'region',
        'days_until_renewal', 'renewal_urgency', 'arr_tier',
        'plan_tier_numeric', 'arr_plan_mismatch', 'in_renewal_window'
    ]

    result = df[columns_to_keep].copy()

    print(f"[Features] Contract features computed for {len(result)} accounts")
    print(f"  In renewal window: {result['in_renewal_window'].sum()}")

    return result


# ============================================================
# 7. CONFLICT FEATURES
# ============================================================

def compute_conflict_features(
    conflicts: List[Dict],
    accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Convert cross-source conflicts into numeric features."""
    features = []
    all_account_ids = accounts_df['account_id'].unique()
    conflict_lookup = {c['account_id']: c for c in conflicts}

    for account_id in all_account_ids:
        row = {'account_id': account_id}

        if account_id in conflict_lookup:
            c = conflict_lookup[account_id]
            row['conflict_count'] = c['conflict_count']
            row['has_conflicts'] = True
            high_severity = [
                cf for cf in c['conflicts']
                if cf.get('severity') == 'high'
            ]
            row['high_severity_conflicts'] = len(high_severity)
            conflict_types = [cf['type'] for cf in c['conflicts']]
            row['has_nps_contradiction'] = 'nps_score_comment_mismatch' in conflict_types
            row['has_nps_csm_mismatch'] = 'nps_vs_csm_sentiment' in conflict_types
            row['has_csm_support_mismatch'] = 'csm_vs_support_tickets' in conflict_types
        else:
            row.update({
                'conflict_count': 0, 'has_conflicts': False,
                'high_severity_conflicts': 0, 'has_nps_contradiction': False,
                'has_nps_csm_mismatch': False, 'has_csm_support_mismatch': False
            })

        features.append(row)

    df = pd.DataFrame(features)

    print(f"[Features] Conflict features computed for {len(df)} accounts")
    print(f"  With conflicts: {df['has_conflicts'].sum()}")

    return df


# ============================================================
# 8. MASTER FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    reconciled_data: Dict[str, Any]
) -> tuple:
    """
    Build the master feature matrix by computing and merging all feature sets.
    Returns (all_features_df, renewal_features_df).
    """
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    accounts_df = reconciled_data['all_accounts']

    print("\n--- Computing Usage Features ---")
    usage_features = compute_usage_features(reconciled_data['usage_metrics'])

    print("\n--- Computing Support Features ---")
    support_features = compute_support_features(
        reconciled_data['support_tickets'], accounts_df
    )

    print("\n--- Computing NPS Features ---")
    nps_features = compute_nps_features(
        reconciled_data['nps_processed'], accounts_df
    )

    print("\n--- Computing CSM Features ---")
    csm_features = compute_csm_features(
        reconciled_data['csm_notes_parsed'], accounts_df
    )

    print("\n--- Computing SDK Features ---")
    sdk_features = compute_sdk_features(reconciled_data['sdk_risks'])

    print("\n--- Computing Contract Features ---")
    contract_features = compute_contract_features(accounts_df)

    print("\n--- Computing Conflict Features ---")
    conflict_features = compute_conflict_features(
        reconciled_data['conflicts'], accounts_df
    )

    print("\n--- Merging All Features ---")
    master = contract_features.copy()

    for feature_df, name in [
        (usage_features, 'usage'),
        (support_features, 'support'),
        (nps_features, 'nps'),
        (csm_features, 'csm'),
        (sdk_features, 'sdk'),
        (conflict_features, 'conflict')
    ]:
        before = len(master.columns)
        master = master.merge(feature_df, on='account_id', how='left')
        after = len(master.columns)
        print(f"  Merged {name}: +{after - before} columns, total={after}")

    # Fill NaN health scores
    health_cols = [c for c in master.columns if c.endswith('_health_score')]
    for col in health_cols:
        null_count = master[col].isna().sum()
        if null_count > 0:
            master[col] = master[col].fillna(50)

    renewal_master = master[master['in_renewal_window'] == True].copy()

    print(f"\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print(f"  Total accounts: {len(master)}")
    print(f"  Renewal window accounts: {len(renewal_master)}")
    print(f"  Total features: {len(master.columns)}")
    print("=" * 60)

    # Save with utf-8-sig for Windows Excel compatibility
    master.to_csv('outputs/all_account_features.csv', index=False, encoding='utf-8-sig')
    renewal_master.to_csv(
        'outputs/renewal_account_features.csv', index=False, encoding='utf-8-sig'
    )
    print("Saved: all_account_features.csv, renewal_account_features.csv")

    return master, renewal_master