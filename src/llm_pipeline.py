"""
LLM Pipeline Module — Deep Analysis with LangChain

Handles:
1. Changelog Impact Analysis — which product changes affect which accounts
2. Silent Churn Pattern Detection — LLM identifies hidden churn signals
3. Cross-Signal Deep Analysis — finds non-obvious insights
"""

import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from src.config import REFERENCE_DATE, RENEWAL_CUTOFF_DATE
from src.utils import get_llm


# ============================================================
# HELPER: Safe JSON write
# ============================================================

def safe_write_json(filepath: str, data: Any):
    """Write JSON with UTF-8 encoding."""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def safe_write_csv(df: pd.DataFrame, filepath: str):
    """Write CSV with UTF-8-sig encoding (Excel friendly on Windows)."""
    df.to_csv(filepath, index=False, encoding='utf-8-sig')


# ============================================================
# 1. CHANGELOG IMPACT ANALYSIS
# ============================================================

def analyze_changelog_impact(
    changelog_raw: str,
    accounts_features: pd.DataFrame,
    csm_notes_parsed: List[Dict],
    support_tickets: pd.DataFrame,
    provider: str = "groq"
) -> pd.DataFrame:
    """
    Analyze how product changelog items specifically impact each renewal account.
    """
    llm = get_llm(provider=provider, temperature=0.0, max_tokens=3000)

    print("[LLM Pipeline] Extracting key changelog items...")

    changelog_extract_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a product analyst. Extract the most impactful items from this changelog
that would affect existing customers — especially deprecations, breaking changes,
removed features, and critical bug fixes. Return valid JSON only. No markdown."""),
        ("human", """Extract impactful changelog items from this product changelog.

For each item identify:
- What changed
- Who is affected (which SDK versions, which plan tiers, which use cases)
- The deadline or date
- The severity (critical/high/medium/low)
- What action customers need to take

CHANGELOG:
{changelog}

Return JSON array:
[
    {{
        "item": "short description",
        "category": "deprecation|breaking_change|bug_fix|new_feature|security",
        "affected_sdk_versions": ["v3.x", "v4.0"],
        "affected_plan_tiers": ["all"],
        "deadline": "YYYY-MM-DD or null",
        "severity": "critical|high|medium|low",
        "customer_action_required": "what they need to do",
        "keywords": ["related", "search", "terms"]
    }}
]""")
    ])

    changelog_chain = changelog_extract_prompt | llm | JsonOutputParser()

    try:
        changelog_items = changelog_chain.invoke({"changelog": changelog_raw})
        print(f"  Extracted {len(changelog_items)} impactful changelog items")
        for item in changelog_items:
            print(f"    - [{item.get('severity', '?')}] {item.get('item', '?')}")
    except Exception as e:
        print(f"  WARNING: Changelog extraction failed: {e}")
        changelog_items = []

    # Match changelog items to accounts
    print("[LLM Pipeline] Matching changelog impacts to accounts...")

    impact_records = []
    renewal_accounts = accounts_features[accounts_features['in_renewal_window'] == True]

    for _, account in renewal_accounts.iterrows():
        account_id = account['account_id']
        account_impacts = []

        sdk_version = str(account.get('sdk_version', '')).strip()
        plan_tier = str(account.get('plan_tier', '')).strip()

        account_tickets = support_tickets[support_tickets['account_id'] == account_id]
        ticket_subjects = account_tickets['subject'].str.lower().tolist() if len(account_tickets) > 0 else []
        ticket_text = ' '.join(ticket_subjects)

        account_notes = [
            n for n in csm_notes_parsed
            if n.get('matched_account_id') == account_id
        ]
        notes_text = ' '.join([
            n.get('summary', '') + ' ' + ' '.join(n.get('churn_risk_signals', []))
            for n in account_notes
        ]).lower()

        for item in changelog_items:
            is_affected = False
            reasons = []

            affected_sdks = item.get('affected_sdk_versions', [])
            if affected_sdks:
                for affected_sdk in affected_sdks:
                    affected_sdk_clean = str(affected_sdk).lower().strip()
                    sdk_clean = sdk_version.lower().strip()

                    if 'v3' in affected_sdk_clean and sdk_clean.startswith('v3'):
                        is_affected = True
                        reasons.append(
                            f"Account on {sdk_version}, affected by {affected_sdk_clean} changes"
                        )

                    if affected_sdk_clean in ['v4.0', 'v4.1'] and sdk_clean.startswith(('v4.0', 'v4.1')):
                        is_affected = True
                        reasons.append(
                            f"Account on {sdk_version}, affected by {affected_sdk_clean} changes"
                        )

            affected_tiers = item.get('affected_plan_tiers', ['all'])
            if 'all' not in [t.lower() for t in affected_tiers]:
                if plan_tier in affected_tiers:
                    is_affected = True
                    reasons.append(f"Account on {plan_tier} plan, which is affected")

            keywords = item.get('keywords', [])
            for kw in keywords:
                if kw.lower() in ticket_text and not is_affected:
                    is_affected = True
                    reasons.append(f"Account has ticket related to '{kw}'")
                    break

            for kw in keywords:
                if kw.lower() in notes_text and not is_affected:
                    is_affected = True
                    reasons.append(f"CSM notes mention '{kw}'")
                    break

            if is_affected:
                account_impacts.append({
                    'changelog_item': item.get('item', ''),
                    'category': item.get('category', ''),
                    'severity': item.get('severity', ''),
                    'deadline': item.get('deadline', ''),
                    'action_required': item.get('customer_action_required', ''),
                    'match_reasons': reasons
                })

        impact_records.append({
            'account_id': account_id,
            'changelog_impacts': account_impacts,
            'changelog_impact_count': len(account_impacts),
            'critical_impacts': len([i for i in account_impacts if i['severity'] == 'critical']),
            'high_impacts': len([i for i in account_impacts if i['severity'] == 'high']),
            'has_deadline_pressure': any(
                i.get('deadline') and i.get('deadline') != 'null'
                for i in account_impacts
            ),
            'changelog_risk_score': min(100, sum(
                {'critical': 30, 'high': 20, 'medium': 10, 'low': 5}.get(i['severity'], 0)
                for i in account_impacts
            ))
        })

    impact_df = pd.DataFrame(impact_records)

    print(f"[LLM Pipeline] Changelog impact analysis complete:")
    print(f"  Accounts with impacts: {(impact_df['changelog_impact_count'] > 0).sum()}")
    print(f"  Accounts with critical impacts: {(impact_df['critical_impacts'] > 0).sum()}")

    return impact_df


# ============================================================
# 2. SILENT CHURN PATTERN DETECTION
# ============================================================

def detect_silent_churn_patterns(
    accounts_features: pd.DataFrame,
    csm_notes_parsed: List[Dict],
    nps_processed: pd.DataFrame,
    provider: str = "groq"
) -> pd.DataFrame:
    """
    Use LLM to detect silent churn patterns per account.
    """
    llm = get_llm(provider=provider, temperature=0.1, max_tokens=2000)

    print("[LLM Pipeline] Detecting silent churn patterns...")

    renewal_accounts = accounts_features[accounts_features['in_renewal_window'] == True]

    pattern_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert customer success analyst specializing in B2B SaaS churn prediction.

You specialize in detecting SILENT CHURN - customers who appear healthy on surface metrics
but are actually at risk. These are accounts that traditional rule-based systems miss.

Silent churn patterns include:
1. HAPPY BUT LEAVING: High NPS/positive sentiment but usage declining
2. BUILDING ALTERNATIVES: Positive engagement but asking about data exports or migration APIs
3. GONE QUIET: Stopped filing tickets, stopped engaging, usage flat or declining
4. SURVEY FATIGUE: Generic/copied NPS responses indicate disengagement
5. PLATFORM STAGNATION: Never upgraded SDK, never adopted new features
6. CHAMPION DEPENDENCY: Only 1-2 active users, if they leave the account churns
7. ORGANIZATIONAL CHANGE: Mergers, acquisitions, leadership changes
8. COMPLIANCE CLIFF: Regulatory requirements the vendor cannot meet

Return valid JSON only. No markdown."""),
        ("human", """Analyze this account for silent churn patterns:

ACCOUNT PROFILE:
- Name: {account_name}
- ARR: ${arr}
- Plan: {plan_tier}
- Renewal in: {days_until_renewal} days
- Industry: {industry}

USAGE SIGNALS:
- API calls trend: {api_trend}% change (recent vs older period)
- Content creation trend: {content_trend}% change
- Active users trend: {active_users_trend}% change
- Active users (recent avg): {active_users_recent}
- Workflows trend: {workflow_trend}% change
- Workflow abandoned: {workflow_abandoned}
- SDK version: {sdk_version}

SUPPORT SIGNALS:
- Total tickets: {total_tickets}
- Open P1 tickets: {open_p1}
- Has blocking tickets: {has_blocking}
- Has deprecation tickets: {has_deprecation}
- No tickets at all: {no_tickets}

NPS SIGNALS:
- Score: {nps_score}
- Category: {nps_category}
- Is contradictory: {nps_contradictory}
- Is generic template: {nps_generic}
- Has response: {nps_has_response}

CSM SIGNALS:
- Sentiment: {csm_sentiment}
- Churn signals count: {csm_churn_signals}
- Competitors mentioned: {csm_competitors}
- Executive involvement: {csm_executive}
- Missed meetings: {csm_missed}
- Has evaluation signals: {csm_evaluation}
- Budget concerns: {csm_budget}

CSM NOTE SUMMARIES:
{csm_summaries}

Return JSON:
{{
    "silent_churn_patterns_detected": [
        {{
            "pattern_name": "name of the pattern",
            "confidence": "high|medium|low",
            "evidence": "specific evidence from the data",
            "why_non_obvious": "why a simple rule-based system would miss this",
            "risk_contribution": 0
        }}
    ],
    "overall_silent_churn_risk": "high|medium|low|none",
    "recommended_investigation": "what the account team should look into"
}}""")
    ])

    pattern_chain = pattern_prompt | llm | JsonOutputParser()

    results = []

    for _, account in renewal_accounts.iterrows():
        account_id = account['account_id']

        account_notes = [
            n for n in csm_notes_parsed
            if n.get('matched_account_id') == account_id
        ]
        csm_summaries = '\n'.join([
            f"- {n.get('summary', 'No summary')}"
            for n in account_notes
        ]) if account_notes else "No CSM notes available"

        nps_row = nps_processed[nps_processed['account_id'] == account_id]

        try:
            result = pattern_chain.invoke({
                "account_name": account.get('account_name', ''),
                "arr": f"{account.get('arr', 0):,.0f}",
                "plan_tier": account.get('plan_tier', ''),
                "days_until_renewal": account.get('days_until_renewal', ''),
                "industry": account.get('industry', ''),
                "api_trend": round(float(account.get('api_calls_pct_change', 0)), 1),
                "content_trend": round(float(account.get('content_entries_created_pct_change', 0)), 1),
                "active_users_trend": round(float(account.get('active_users_pct_change', 0)), 1),
                "active_users_recent": round(float(account.get('active_users_recent_avg', 0)), 1),
                "workflow_trend": round(float(account.get('workflows_triggered_pct_change', 0)), 1),
                "workflow_abandoned": bool(account.get('workflow_abandoned', False)),
                "sdk_version": account.get('sdk_version', 'unknown'),
                "total_tickets": int(account.get('total_tickets', 0)),
                "open_p1": int(account.get('open_p1_tickets', 0)),
                "has_blocking": bool(account.get('has_blocking_tickets', False)),
                "has_deprecation": bool(account.get('has_deprecation_tickets', False)),
                "no_tickets": bool(account.get('no_tickets_flag', False)),
                "nps_score": int(nps_row['score'].values[0]) if len(nps_row) > 0 else 'N/A',
                "nps_category": str(account.get('nps_category', 'no_response')),
                "nps_contradictory": bool(account.get('nps_is_contradictory', False)),
                "nps_generic": bool(account.get('nps_is_generic', False)),
                "nps_has_response": bool(account.get('nps_has_response', False)),
                "csm_sentiment": str(account.get('csm_sentiment', 'unknown')),
                "csm_churn_signals": int(account.get('csm_churn_signal_count', 0)),
                "csm_competitors": str(account.get('csm_competitors_list', '')),
                "csm_executive": bool(account.get('csm_has_executive_involvement', False)),
                "csm_missed": bool(account.get('csm_has_missed_meetings', False)),
                "csm_evaluation": bool(account.get('csm_has_evaluation_signals', False)),
                "csm_budget": bool(account.get('csm_has_budget_concerns', False)),
                "csm_summaries": csm_summaries
            })

            patterns = result.get('silent_churn_patterns_detected', [])
            silent_risk = result.get('overall_silent_churn_risk', 'none')
            investigation = result.get('recommended_investigation', '')

            silent_score = sum(
                p.get('risk_contribution', 0) for p in patterns
            )

            results.append({
                'account_id': account_id,
                'silent_churn_patterns': patterns,
                'silent_churn_pattern_count': len(patterns),
                'silent_churn_risk': silent_risk,
                'silent_churn_score': min(100, silent_score),
                'recommended_investigation': investigation
            })

            if patterns:
                print(f"  [{account_id}] {account.get('account_name', '')}: "
                      f"{len(patterns)} patterns (risk={silent_risk})")

        except Exception as e:
            print(f"  WARNING: Silent churn analysis failed for {account_id}: {e}")
            results.append({
                'account_id': account_id,
                'silent_churn_patterns': [],
                'silent_churn_pattern_count': 0,
                'silent_churn_risk': 'unknown',
                'silent_churn_score': 0,
                'recommended_investigation': f'Analysis failed: {str(e)}'
            })

        time.sleep(0.5)

    result_df = pd.DataFrame(results)

    print(f"\n[LLM Pipeline] Silent churn analysis complete:")
    print(f"  Accounts analyzed: {len(result_df)}")
    print(f"  High silent churn risk: {(result_df['silent_churn_risk'] == 'high').sum()}")
    print(f"  Medium silent churn risk: {(result_df['silent_churn_risk'] == 'medium').sum()}")

    return result_df


# ============================================================
# 3. PORTFOLIO-LEVEL INSIGHTS
# ============================================================

def generate_cross_signal_insights(
    accounts_features: pd.DataFrame,
    csm_notes_parsed: List[Dict],
    nps_processed: pd.DataFrame,
    changelog_impacts: pd.DataFrame,
    silent_churn: pd.DataFrame,
    provider: str = "groq"
) -> Dict[str, Any]:
    """
    Generate portfolio-level non-obvious insights.
    """
    llm = get_llm(provider=provider, temperature=0.2, max_tokens=4000)

    print("[LLM Pipeline] Generating cross-signal portfolio insights...")

    renewal_accounts = accounts_features[accounts_features['in_renewal_window'] == True]

    portfolio_summary = {
        "total_renewal_accounts": int(len(renewal_accounts)),
        "total_renewal_arr": f"${renewal_accounts['arr'].sum():,.0f}",
        "avg_usage_health": f"{renewal_accounts['usage_health_score'].mean():.1f}",
        "avg_support_health": f"{renewal_accounts['support_health_score'].mean():.1f}",
        "accounts_on_deprecated_sdk": int(
            renewal_accounts.get('sdk_is_v3', pd.Series(
                [False] * len(renewal_accounts)
            )).sum()
        ),
        "accounts_with_declining_usage": int(
            renewal_accounts.get(
                'api_calls_declining',
                pd.Series([False] * len(renewal_accounts))
            ).sum()
        ),
        "accounts_with_competitors": int(
            (renewal_accounts.get(
                'csm_competitor_count',
                pd.Series([0] * len(renewal_accounts))
            ) > 0).sum()
        ),
    }

    csm_analysis = {}
    if 'csm_name' in renewal_accounts.columns:
        for csm, group in renewal_accounts.groupby('csm_name'):
            declining = group.get(
                'api_calls_declining',
                pd.Series([False] * len(group))
            )
            csm_analysis[str(csm)] = {
                "account_count": int(len(group)),
                "total_arr": f"${group['arr'].sum():,.0f}",
                "avg_usage_health": f"{group['usage_health_score'].mean():.1f}",
                "declining_accounts": int(declining.sum()),
            }

    industry_analysis = {}
    if 'industry' in renewal_accounts.columns:
        for industry, group in renewal_accounts.groupby('industry'):
            industry_analysis[str(industry)] = {
                "account_count": int(len(group)),
                "avg_usage_health": f"{group['usage_health_score'].mean():.1f}",
                "avg_nps_health": f"{group['nps_health_score'].mean():.1f}",
            }

    high_silent = silent_churn[silent_churn['silent_churn_risk'] == 'high']
    silent_summary = {
        "high_risk_count": int(len(high_silent)),
        "high_risk_accounts": [int(x) for x in high_silent['account_id'].tolist()],
        "total_patterns_detected": int(silent_churn['silent_churn_pattern_count'].sum())
    }

    insight_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior BizOps strategist at a B2B SaaS company.
You excel at finding NON-OBVIOUS, ACTIONABLE insights that leadership would miss.

Focus on insights that:
1. Span multiple accounts (systemic, not individual)
2. Combine signals from different data sources in unexpected ways
3. Have concrete business impact (quantify ARR at risk when possible)
4. Are actionable with specific interventions
5. Would surprise the executive team

Return valid JSON only. No markdown."""),
        ("human", """Analyze this portfolio of renewal accounts and find non-obvious insights.

PORTFOLIO SUMMARY:
{portfolio_summary}

CSM WORKLOAD:
{csm_analysis}

INDUSTRY BREAKDOWN:
{industry_analysis}

SILENT CHURN PATTERNS:
{silent_summary}

CHANGELOG CONTEXT:
- SDK v3.x sunset date: April 30, 2026
- Legacy editor removal: May 2026 (v4.4.0)
- Legacy workflow engine deprecated: cannot edit after Feb 28, 2026
- REST API v2 sunset: April 30, 2026

Generate 3-5 non-obvious portfolio-level insights.

Return JSON:
{{
    "portfolio_insights": [
        {{
            "insight_title": "short title",
            "insight_detail": "detailed explanation (2-3 sentences)",
            "evidence": "specific data points supporting this",
            "arr_at_risk": "estimated ARR at risk",
            "recommended_action": "specific action to take",
            "urgency": "immediate|this_week|this_month|this_quarter",
            "insight_type": "systemic_risk|csm_pattern|industry_cluster|timing_risk|product_gap|silent_churn"
        }}
    ],
    "top_priority_action": "the single most important thing to do this week"
}}""")
    ])

    insight_chain = insight_prompt | llm | JsonOutputParser()

    try:
        insights = insight_chain.invoke({
            "portfolio_summary": json.dumps(portfolio_summary, indent=2),
            "csm_analysis": json.dumps(csm_analysis, indent=2),
            "industry_analysis": json.dumps(industry_analysis, indent=2),
            "silent_summary": json.dumps(silent_summary, indent=2)
        })

        print(f"  Generated {len(insights.get('portfolio_insights', []))} portfolio insights")
        for insight in insights.get('portfolio_insights', []):
            print(f"    - [{insight.get('urgency', '?')}] {insight.get('insight_title', '?')}")

    except Exception as e:
        print(f"  WARNING: Portfolio insight generation failed: {e}")
        insights = {
            "portfolio_insights": [],
            "top_priority_action": f"Analysis failed: {str(e)}"
        }

    return insights


# ============================================================
# 4. MASTER LLM PIPELINE
# ============================================================

def run_llm_pipeline(
    reconciled_data: Dict[str, Any],
    feature_matrix: pd.DataFrame,
    provider: str = "groq"
) -> Dict[str, Any]:
    """
    Master LLM pipeline — runs all deep analysis steps.
    """
    print("\n" + "=" * 60)
    print("LLM DEEP ANALYSIS PIPELINE")
    print("=" * 60)

    print("\n[Step 1] Analyzing changelog impact on accounts...")
    changelog_impacts = analyze_changelog_impact(
        changelog_raw=reconciled_data['changelog_raw'],
        accounts_features=feature_matrix,
        csm_notes_parsed=reconciled_data['csm_notes_parsed'],
        support_tickets=reconciled_data['support_tickets'],
        provider=provider
    )

    print("\n[Step 2] Detecting silent churn patterns...")
    silent_churn = detect_silent_churn_patterns(
        accounts_features=feature_matrix,
        csm_notes_parsed=reconciled_data['csm_notes_parsed'],
        nps_processed=reconciled_data['nps_processed'],
        provider=provider
    )

    print("\n[Step 3] Generating portfolio-level insights...")
    portfolio_insights = generate_cross_signal_insights(
        accounts_features=feature_matrix,
        csm_notes_parsed=reconciled_data['csm_notes_parsed'],
        nps_processed=reconciled_data['nps_processed'],
        changelog_impacts=changelog_impacts,
        silent_churn=silent_churn,
        provider=provider
    )

    # Save all outputs with proper encoding
    safe_write_csv(changelog_impacts, 'outputs/changelog_impacts.csv')

    silent_export = silent_churn.copy()
    silent_export['silent_churn_patterns'] = silent_export['silent_churn_patterns'].apply(
        lambda x: json.dumps(x, ensure_ascii=False)
    )
    safe_write_csv(silent_export, 'outputs/silent_churn_analysis.csv')

    safe_write_json('outputs/portfolio_insights.json', portfolio_insights)

    llm_results = {
        'changelog_impacts': changelog_impacts,
        'silent_churn': silent_churn,
        'portfolio_insights': portfolio_insights
    }

    print("\n" + "=" * 60)
    print("LLM PIPELINE COMPLETE")
    print(f"  Changelog impacts: {len(changelog_impacts)}")
    print(f"  Silent churn patterns: {silent_churn['silent_churn_pattern_count'].sum()}")
    print(f"  Portfolio insights: {len(portfolio_insights.get('portfolio_insights', []))}")
    print("=" * 60)

    return llm_results