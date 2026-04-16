"""
Risk Scoring Module — LangGraph Stateful Risk Assessment

Node names use 'node_' prefix to avoid collision with state key names.
State fields use 'computed_' prefix for pipeline outputs.
"""

import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, TypedDict

from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from src.config import REFERENCE_DATE, RENEWAL_CUTOFF_DATE, RISK_TIER_HIGH, RISK_TIER_MEDIUM
from src.utils import get_llm


# ============================================================
# 1. STATE DEFINITION
# ============================================================

class AccountRiskState(TypedDict):
    """State object flowing through the LangGraph pipeline."""
    # Input data
    account_id: int
    account_name: str
    arr: float
    plan_tier: str
    industry: str
    region: str
    csm_name: str
    days_until_renewal: int
    renewal_urgency: str

    # Health scores
    usage_health_score: float
    support_health_score: float
    nps_health_score: float
    csm_health_score: float
    sdk_health_score: float

    # Key signal flags
    key_signals: Dict[str, Any]

    # LLM pipeline outputs
    changelog_risk_score: float
    silent_churn_score: float
    silent_churn_patterns: List[Dict]
    changelog_impacts: List[Dict]

    # Conflicts
    conflicts: List[Dict]
    conflict_count: int

    # CSM / NPS details
    csm_summaries: List[str]
    csm_competitors: str
    nps_score: Optional[int]
    nps_category: str
    nps_is_contradictory: bool

    # Computed by pipeline nodes — prefixed with 'computed_' to avoid name collision
    computed_quant_score: float
    computed_qual_assessment: Dict[str, Any]
    computed_conflict_resolution: Dict[str, Any]
    computed_final_risk_score: float
    computed_risk_tier: str
    computed_confidence: str
    computed_risk_factors: List[Dict]
    computed_actions: List[str]


# ============================================================
# 2. NODE FUNCTIONS
# ============================================================

def node_compute_quantitative(state: AccountRiskState) -> dict:
    """Node 1: Weighted quantitative risk score."""
    weights = {
        'usage': 0.25, 'csm': 0.25, 'support': 0.15,
        'nps': 0.10, 'sdk': 0.10, 'changelog': 0.08, 'silent_churn': 0.07
    }

    usage_health = float(state.get('usage_health_score', 50))
    support_health = float(state.get('support_health_score', 50))
    nps_health = float(state.get('nps_health_score', 50))
    csm_health = float(state.get('csm_health_score', 50))
    sdk_health = float(state.get('sdk_health_score', 50))
    changelog_health = max(0.0, 100.0 - float(state.get('changelog_risk_score', 0)))
    silent_health = max(0.0, 100.0 - float(state.get('silent_churn_score', 0)))

    weighted_health = (
        weights['usage'] * usage_health +
        weights['csm'] * csm_health +
        weights['support'] * support_health +
        weights['nps'] * nps_health +
        weights['sdk'] * sdk_health +
        weights['changelog'] * changelog_health +
        weights['silent_churn'] * silent_health
    )

    days = int(state.get('days_until_renewal', 90))
    if days <= 15:
        urgency_multiplier = 1.3
    elif days <= 30:
        urgency_multiplier = 1.2
    elif days <= 45:
        urgency_multiplier = 1.1
    else:
        urgency_multiplier = 1.0

    arr = float(state.get('arr', 0))
    if arr >= 1_000_000:
        arr_multiplier = 1.15
    elif arr >= 500_000:
        arr_multiplier = 1.10
    elif arr >= 100_000:
        arr_multiplier = 1.05
    else:
        arr_multiplier = 1.0

    raw_risk = 100.0 - weighted_health
    adjusted_risk = min(100.0, raw_risk * urgency_multiplier * arr_multiplier)

    risk_factors = []

    def add_factor(name, health, weight, low_threshold=40, med_threshold=60):
        if health < low_threshold:
            risk_factors.append({
                'factor': name, 'severity': 'high',
                'score_impact': round((50 - health) * weight, 1),
                'detail': f'{name} health score: {health:.0f}/100'
            })
        elif health < med_threshold:
            risk_factors.append({
                'factor': name, 'severity': 'medium',
                'score_impact': round((50 - health) * weight, 1),
                'detail': f'{name} health score: {health:.0f}/100'
            })

    add_factor('Usage Decline', usage_health, weights['usage'])
    add_factor('CSM Sentiment', csm_health, weights['csm'])
    add_factor('Support Issues', support_health, weights['support'])

    if nps_health < 40:
        risk_factors.append({
            'factor': 'NPS Score',
            'severity': 'high' if nps_health < 20 else 'medium',
            'score_impact': round((50 - nps_health) * weights['nps'], 1),
            'detail': f'NPS health score: {nps_health:.0f}/100'
        })

    if sdk_health < 30:
        risk_factors.append({
            'factor': 'SDK/Platform Risk', 'severity': 'critical',
            'score_impact': round((50 - sdk_health) * weights['sdk'], 1),
            'detail': f'SDK health score: {sdk_health:.0f}/100'
        })

    cl_score = float(state.get('changelog_risk_score', 0))
    if cl_score > 30:
        risk_factors.append({
            'factor': 'Product Change Impact',
            'severity': 'high' if cl_score > 50 else 'medium',
            'score_impact': round(cl_score * weights['changelog'], 1),
            'detail': f'Changelog risk: {cl_score:.0f}/100'
        })

    sc_score = float(state.get('silent_churn_score', 0))
    if sc_score > 20:
        risk_factors.append({
            'factor': 'Silent Churn Pattern',
            'severity': 'high' if sc_score > 50 else 'medium',
            'score_impact': round(sc_score * weights['silent_churn'], 1),
            'detail': f'Silent churn score: {sc_score:.0f}/100'
        })

    risk_factors.sort(key=lambda x: x['score_impact'], reverse=True)

    return {
        'computed_quant_score': round(adjusted_risk, 1),
        'computed_risk_factors': risk_factors
    }


def node_qualitative_assessment(state: AccountRiskState) -> dict:
    """Node 2: LLM qualitative adjustment."""
    llm = get_llm(provider="groq", temperature=0.1, max_tokens=1500)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior customer success strategist.
Assess the QUALITATIVE renewal risk for this account.
Focus on factors that numbers cannot capture.
Return valid JSON only. No markdown."""),
        ("human", """Assess qualitative renewal risk:

ACCOUNT: {account_name} (ID: {account_id})
ARR: ${arr} | Plan: {plan_tier} | Renewal in: {days} days
QUANTITATIVE RISK SCORE: {quant_score}/100

KEY RISK FACTORS:
{risk_factors}

CSM NOTES:
{csm_summaries}

COMPETITORS MENTIONED: {competitors}

SILENT CHURN PATTERNS:
{silent_patterns}

DATA SOURCE CONFLICTS:
{conflicts}

NPS: {nps_score} ({nps_category}) | Contradictory: {nps_contradictory}

Return JSON:
{{
    "qualitative_risk_adjustment": integer from -15 to 15,
    "adjustment_reason": "one sentence",
    "key_qualitative_factors": ["factor1", "factor2"],
    "relationship_health": "strong|stable|strained|critical",
    "executive_attention_needed": true or false,
    "immediate_action_needed": true or false
}}""")
    ])

    chain = prompt | llm | JsonOutputParser()

    risk_factors_text = '\n'.join([
        f"- [{f['severity']}] {f['factor']}: {f['detail']}"
        for f in state.get('computed_risk_factors', [])
    ]) or "None identified"

    csm_text = '\n'.join(state.get('csm_summaries', [])) or "No CSM notes available"

    silent_text = '\n'.join([
        f"- {p.get('pattern_name', '?')}: {p.get('evidence', '?')}"
        for p in state.get('silent_churn_patterns', [])
    ]) or "None detected"

    conflicts_text = '\n'.join([
        f"- [{c.get('severity', '?')}] {c.get('type', '?')}: {c.get('detail', '?')}"
        for c in state.get('conflicts', [])
    ]) or "No conflicts"

    try:
        result = chain.invoke({
            "account_name": state.get('account_name', ''),
            "account_id": state.get('account_id', ''),
            "arr": f"{state.get('arr', 0):,.0f}",
            "plan_tier": state.get('plan_tier', ''),
            "days": state.get('days_until_renewal', 0),
            "quant_score": state.get('computed_quant_score', 0),
            "risk_factors": risk_factors_text,
            "csm_summaries": csm_text,
            "competitors": state.get('csm_competitors', 'None'),
            "silent_patterns": silent_text,
            "conflicts": conflicts_text,
            "nps_score": state.get('nps_score', 'N/A'),
            "nps_category": state.get('nps_category', 'unknown'),
            "nps_contradictory": state.get('nps_is_contradictory', False)
        })
        return {'computed_qual_assessment': result}

    except Exception as e:
        print(f"  WARNING: Qualitative assessment failed for {state.get('account_id')}: {e}")
        return {
            'computed_qual_assessment': {
                'qualitative_risk_adjustment': 0,
                'adjustment_reason': f'Assessment failed: {str(e)}',
                'key_qualitative_factors': [],
                'relationship_health': 'unknown',
                'executive_attention_needed': False,
                'immediate_action_needed': False
            }
        }


def node_resolve_conflicts(state: AccountRiskState) -> dict:
    """Node 3: Rule-based conflict resolution."""
    conflicts = state.get('conflicts', [])

    if not conflicts:
        return {
            'computed_conflict_resolution': {
                'has_conflicts': False,
                'resolution_adjustment': 0,
                'resolution_notes': 'No conflicts to resolve',
                'trusted_signals': [],
                'discounted_signals': []
            }
        }

    adjustment = 0
    trusted = []
    discounted = []
    notes = []

    for conflict in conflicts:
        conflict_type = conflict.get('type', '')
        severity = conflict.get('severity', 'low')

        if conflict_type == 'nps_score_comment_mismatch':
            adjustment += 5
            discounted.append('nps_score')
            trusted.append('nps_comment_sentiment')
            notes.append('NPS score contradicts comment — trusting more negative signal.')

        elif conflict_type == 'nps_vs_csm_sentiment':
            adjustment += 8 if severity == 'high' else 3
            trusted.append('csm_notes')
            discounted.append('nps_score')
            notes.append('NPS and CSM sentiment disagree — trusting CSM notes.')

        elif conflict_type == 'csm_vs_support_tickets':
            adjustment += 5
            trusted.append('support_tickets')
            discounted.append('csm_sentiment')
            notes.append('CSM positive but support tickets show problems — trusting tickets.')

        elif conflict_type == 'generic_nps_extreme_score':
            adjustment += 3
            discounted.append('nps_comment')
            notes.append('Generic NPS comment with extreme score — low reliability.')

    return {
        'computed_conflict_resolution': {
            'has_conflicts': True,
            'resolution_adjustment': adjustment,
            'resolution_notes': ' | '.join(notes),
            'trusted_signals': list(set(trusted)),
            'discounted_signals': list(set(discounted))
        }
    }


def _generate_actions(state: AccountRiskState, tier: str, score: float) -> List[str]:
    """Generate recommended actions based on risk profile."""
    actions = []
    key_signals = state.get('key_signals', {})
    days = int(state.get('days_until_renewal', 90))

    if days <= 15 and tier in ['High', 'Medium']:
        actions.append(
            f"URGENT: Renewal in {days} days. Schedule executive-level call this week."
        )
    elif days <= 30 and tier == 'High':
        actions.append(
            "Schedule renewal conversation this week. Involve sales leadership."
        )

    competitors = state.get('csm_competitors', '')
    if competitors:
        actions.append(
            f"Competitor threat ({competitors}): Prepare competitive battle card and "
            f"schedule value demonstration."
        )

    if key_signals.get('sdk_is_critical', False):
        actions.append(
            "SDK v3.x sunset approaching: Offer dedicated SA migration support."
        )

    open_p1 = key_signals.get('open_p1_tickets', 0)
    if open_p1 > 0:
        actions.append(
            f"{open_p1} open P1 ticket(s): Escalate to engineering. "
            f"Resolve before renewal conversation."
        )

    if key_signals.get('api_calls_severe_decline', False):
        actions.append(
            "Severe usage decline: Schedule product adoption review. "
            "Consider training or professional services."
        )

    if key_signals.get('csm_has_budget_concerns', False):
        actions.append(
            "Budget concerns flagged: Prepare flexible pricing options. "
            "Consider multi-year discount."
        )

    if key_signals.get('csm_has_compliance_blockers', False):
        actions.append(
            "Compliance blocker: Fast-track security questionnaire. "
            "Engage compliance team immediately."
        )

    qual = state.get('computed_qual_assessment', {})
    if qual.get('executive_attention_needed', False):
        actions.append("Executive attention needed: Loop in VP of CS or CRO.")

    if key_signals.get('workflow_abandoned', False):
        actions.append(
            "Workflow abandonment: Investigate if Feb update broke automation. "
            "Offer migration assistance."
        )

    if key_signals.get('csm_has_missed_meetings', False):
        actions.append(
            "Multiple missed meetings: Try alternative contacts. "
            "Primary contact may have changed roles."
        )

    if state.get('nps_category') == 'no_response':
        actions.append(
            "No NPS response — account may be disengaged. "
            "Reach out for informal check-in before formal renewal."
        )

    if not actions:
        if tier == 'Low':
            actions.append(
                "Low risk: Proceed with standard renewal. Consider expansion conversation."
            )
        else:
            actions.append("Schedule proactive check-in before renewal.")

    return actions


def node_assign_final_risk(state: AccountRiskState) -> dict:
    """Node 4: Combine scores, apply overrides, assign tier."""
    quant_score = float(state.get('computed_quant_score', 50))

    qual = state.get('computed_qual_assessment', {})
    qual_adjustment = max(-15, min(15, float(qual.get('qualitative_risk_adjustment', 0))))

    conflict_res = state.get('computed_conflict_resolution', {})
    conflict_adjustment = float(conflict_res.get('resolution_adjustment', 0))

    final_score = min(100.0, max(0.0, quant_score + qual_adjustment + conflict_adjustment))

    if final_score >= RISK_TIER_HIGH:
        tier = 'High'
    elif final_score >= RISK_TIER_MEDIUM:
        tier = 'Medium'
    else:
        tier = 'Low'

    key_signals = state.get('key_signals', {})
    force_high = False

    if (key_signals.get('csm_has_evaluation_signals', False) and
            key_signals.get('csm_sentiment') == 'negative'):
        force_high = True

    if (key_signals.get('sdk_is_critical', False) and
            int(state.get('days_until_renewal', 90)) <= 30):
        force_high = True

    if key_signals.get('csm_has_compliance_blockers', False):
        force_high = True

    if force_high and tier != 'High':
        tier = 'High'
        final_score = max(final_score, float(RISK_TIER_HIGH))

    actions = _generate_actions(state, tier, final_score)

    return {
        'computed_final_risk_score': round(final_score, 1),
        'computed_risk_tier': tier,
        'computed_actions': actions
    }


def node_calibrate_confidence(state: AccountRiskState) -> dict:
    """Node 5: Assess confidence in the risk assessment."""
    confidence_score = 100

    if state.get('nps_score') is None:
        confidence_score -= 15

    if not state.get('csm_summaries', []):
        confidence_score -= 20

    conflict_count = int(state.get('conflict_count', 0))
    if conflict_count > 0:
        confidence_score -= min(conflict_count * 10, 25)

    if state.get('nps_is_contradictory', False):
        confidence_score -= 10

    if state.get('key_signals', {}).get('nps_is_generic', False):
        confidence_score -= 10

    confidence_score = max(0, confidence_score)

    if confidence_score >= 75:
        label = 'high'
    elif confidence_score >= 50:
        label = 'medium'
    else:
        label = 'low'

    return {'computed_confidence': label}


# ============================================================
# 3. BUILD LANGGRAPH
# ============================================================

def build_risk_assessment_graph():
    """
    Build the LangGraph state machine.
    Node names use 'node_' prefix — must NOT match any state key name.
    """
    workflow = StateGraph(AccountRiskState)

    workflow.add_node("node_quantitative", node_compute_quantitative)
    workflow.add_node("node_qualitative", node_qualitative_assessment)
    workflow.add_node("node_conflicts", node_resolve_conflicts)
    workflow.add_node("node_final_risk", node_assign_final_risk)
    workflow.add_node("node_confidence", node_calibrate_confidence)

    workflow.set_entry_point("node_quantitative")
    workflow.add_edge("node_quantitative", "node_qualitative")
    workflow.add_edge("node_qualitative", "node_conflicts")
    workflow.add_edge("node_conflicts", "node_final_risk")
    workflow.add_edge("node_final_risk", "node_confidence")
    workflow.add_edge("node_confidence", END)

    return workflow.compile()


# ============================================================
# 4. PREPARE STATE
# ============================================================

def prepare_account_state(
    account_row: pd.Series,
    csm_notes_parsed: List[Dict],
    conflicts: List[Dict],
    changelog_impacts: pd.DataFrame,
    silent_churn: pd.DataFrame
) -> AccountRiskState:
    """Build initial state for one account."""
    account_id = int(account_row['account_id'])

    account_notes = [
        n for n in csm_notes_parsed
        if n.get('matched_account_id') == account_id
    ]
    csm_summaries = [n.get('summary', 'No summary') for n in account_notes]

    account_conflicts = []
    for c in conflicts:
        if c.get('account_id') == account_id:
            account_conflicts = c.get('conflicts', [])
            break

    cl_row = changelog_impacts[changelog_impacts['account_id'] == account_id]
    cl_risk = float(cl_row['changelog_risk_score'].values[0]) if len(cl_row) > 0 else 0.0
    cl_impacts = []
    if len(cl_row) > 0:
        raw = cl_row['changelog_impacts'].values[0]
        if isinstance(raw, list):
            cl_impacts = raw
        elif isinstance(raw, str):
            try:
                cl_impacts = json.loads(raw)
            except Exception:
                cl_impacts = []

    sc_row = silent_churn[silent_churn['account_id'] == account_id]
    sc_score = float(sc_row['silent_churn_score'].values[0]) if len(sc_row) > 0 else 0.0
    sc_patterns = []
    if len(sc_row) > 0:
        raw = sc_row['silent_churn_patterns'].values[0]
        if isinstance(raw, list):
            sc_patterns = raw
        elif isinstance(raw, str):
            try:
                sc_patterns = json.loads(raw)
            except Exception:
                sc_patterns = []

    key_signals = {
        'api_calls_declining': bool(account_row.get('api_calls_declining', False)),
        'api_calls_severe_decline': bool(account_row.get('api_calls_severe_decline', False)),
        'active_users_declining': bool(account_row.get('active_users_declining', False)),
        'active_users_severe_decline': bool(account_row.get('active_users_severe_decline', False)),
        'workflow_abandoned': bool(account_row.get('workflow_abandoned', False)),
        'declining_metrics_count': int(account_row.get('declining_metrics_count', 0)),
        'open_p1_tickets': int(account_row.get('open_p1_tickets', 0)),
        'escalated_tickets': int(account_row.get('escalated_tickets', 0)),
        'has_blocking_tickets': bool(account_row.get('has_blocking_tickets', False)),
        'has_deprecation_tickets': bool(account_row.get('has_deprecation_tickets', False)),
        'has_recurring_tickets': bool(account_row.get('has_recurring_tickets', False)),
        'no_tickets_flag': bool(account_row.get('no_tickets_flag', False)),
        'sdk_is_critical': bool(account_row.get('sdk_is_critical', False)),
        'sdk_is_v3': bool(account_row.get('sdk_is_v3', False)),
        'nps_is_contradictory': bool(account_row.get('nps_is_contradictory', False)),
        'nps_is_generic': bool(account_row.get('nps_is_generic', False)),
        'csm_sentiment': str(account_row.get('csm_sentiment', 'unknown')),
        'csm_has_evaluation_signals': bool(account_row.get('csm_has_evaluation_signals', False)),
        'csm_has_executive_involvement': bool(account_row.get('csm_has_executive_involvement', False)),
        'csm_has_missed_meetings': bool(account_row.get('csm_has_missed_meetings', False)),
        'csm_has_budget_concerns': bool(account_row.get('csm_has_budget_concerns', False)),
        'csm_has_compliance_blockers': bool(account_row.get('csm_has_compliance_blockers', False)),
        'csm_competitor_count': int(account_row.get('csm_competitor_count', 0)),
    }

    nps_raw = account_row.get('nps_score')
    nps_score = int(nps_raw) if pd.notna(nps_raw) else None

    return AccountRiskState(
        account_id=account_id,
        account_name=str(account_row.get('account_name', '')),
        arr=float(account_row.get('arr', 0)),
        plan_tier=str(account_row.get('plan_tier', '')),
        industry=str(account_row.get('industry', '')),
        region=str(account_row.get('region', '')),
        csm_name=str(account_row.get('csm_name', '')),
        days_until_renewal=int(account_row.get('days_until_renewal', 90)),
        renewal_urgency=str(account_row.get('renewal_urgency', 'upcoming')),
        usage_health_score=float(account_row.get('usage_health_score', 50)),
        support_health_score=float(account_row.get('support_health_score', 50)),
        nps_health_score=float(account_row.get('nps_health_score', 50)),
        csm_health_score=float(account_row.get('csm_health_score', 50)),
        sdk_health_score=float(account_row.get('sdk_health_score', 50)),
        key_signals=key_signals,
        changelog_risk_score=cl_risk,
        silent_churn_score=sc_score,
        silent_churn_patterns=sc_patterns,
        changelog_impacts=cl_impacts,
        conflicts=account_conflicts,
        conflict_count=len(account_conflicts),
        csm_summaries=csm_summaries,
        csm_competitors=str(account_row.get('csm_competitors_list', '')),
        nps_score=nps_score,
        nps_category=str(account_row.get('nps_category', 'no_response')),
        nps_is_contradictory=bool(account_row.get('nps_is_contradictory', False)),
        computed_quant_score=0.0,
        computed_qual_assessment={},
        computed_conflict_resolution={},
        computed_final_risk_score=0.0,
        computed_risk_tier='',
        computed_confidence='',
        computed_risk_factors=[],
        computed_actions=[]
    )


# ============================================================
# 5. MASTER SCORING PIPELINE
# ============================================================

def score_all_accounts(
    feature_matrix: pd.DataFrame,
    reconciled_data: Dict[str, Any],
    llm_results: Dict[str, Any]
) -> pd.DataFrame:
    """Score all renewal accounts through the LangGraph pipeline."""
    print("\n" + "=" * 60)
    print("RISK SCORING — LangGraph Pipeline")
    print("=" * 60)

    graph = build_risk_assessment_graph()
    print("[Risk Scoring] LangGraph compiled successfully")

    renewal_accounts = feature_matrix[feature_matrix['in_renewal_window'] == True]
    print(f"[Risk Scoring] Processing {len(renewal_accounts)} renewal accounts")

    csm_notes = reconciled_data['csm_notes_parsed']
    conflicts = reconciled_data['conflicts']
    changelog_impacts = llm_results['changelog_impacts']
    silent_churn = llm_results['silent_churn']

    results = []

    for idx, (_, account_row) in enumerate(renewal_accounts.iterrows()):
        account_id = int(account_row['account_id'])
        account_name = str(account_row['account_name'])

        print(f"\n  [{idx+1}/{len(renewal_accounts)}] {account_name} ({account_id})...")

        initial_state = prepare_account_state(
            account_row=account_row,
            csm_notes_parsed=csm_notes,
            conflicts=conflicts,
            changelog_impacts=changelog_impacts,
            silent_churn=silent_churn
        )

        try:
            final_state = graph.invoke(initial_state)

            nps_raw = account_row.get('nps_score')
            nps_val = int(nps_raw) if pd.notna(nps_raw) else None

            result = {
                'account_id': account_id,
                'account_name': account_name,
                'arr': float(account_row['arr']),
                'plan_tier': str(account_row['plan_tier']),
                'industry': str(account_row['industry']),
                'region': str(account_row['region']),
                'csm_name': str(account_row['csm_name']),
                'contract_end_date': str(account_row['contract_end_date']),
                'days_until_renewal': int(account_row['days_until_renewal']),
                'renewal_urgency': str(account_row.get('renewal_urgency', '')),
                'quantitative_score': final_state.get('computed_quant_score', 0),
                'qualitative_adjustment': final_state.get(
                    'computed_qual_assessment', {}
                ).get('qualitative_risk_adjustment', 0),
                'conflict_adjustment': final_state.get(
                    'computed_conflict_resolution', {}
                ).get('resolution_adjustment', 0),
                'final_risk_score': final_state.get('computed_final_risk_score', 0),
                'risk_tier': final_state.get('computed_risk_tier', 'Unknown'),
                'confidence': final_state.get('computed_confidence', 'unknown'),
                'usage_health': float(account_row.get('usage_health_score', 50)),
                'support_health': float(account_row.get('support_health_score', 50)),
                'nps_health': float(account_row.get('nps_health_score', 50)),
                'csm_health': float(account_row.get('csm_health_score', 50)),
                'sdk_health': float(account_row.get('sdk_health_score', 50)),
                'nps_score': nps_val,
                'nps_category': str(account_row.get('nps_category', 'unknown')),
                'relationship_health': final_state.get(
                    'computed_qual_assessment', {}
                ).get('relationship_health', 'unknown'),
                'executive_attention': final_state.get(
                    'computed_qual_assessment', {}
                ).get('executive_attention_needed', False),
                'risk_factors': json.dumps(
                    final_state.get('computed_risk_factors', []),
                    ensure_ascii=False
                ),
                'recommended_actions': json.dumps(
                    final_state.get('computed_actions', []),
                    ensure_ascii=False
                ),
                'qualitative_factors': json.dumps(
                    final_state.get('computed_qual_assessment', {}).get(
                        'key_qualitative_factors', []
                    ),
                    ensure_ascii=False
                ),
                'conflict_notes': final_state.get(
                    'computed_conflict_resolution', {}
                ).get('resolution_notes', ''),
                'silent_churn_patterns': json.dumps(
                    final_state.get('silent_churn_patterns', []),
                    ensure_ascii=False
                ),
                'csm_summaries': json.dumps(
                    final_state.get('csm_summaries', []),
                    ensure_ascii=False
                ),
                'competitors': str(final_state.get('csm_competitors', '')),
            }

            tier = result['risk_tier']
            tier_label = {'High': '[HIGH]', 'Medium': '[MED]', 'Low': '[LOW]'}.get(tier, '[?]')
            print(f"    {tier_label} Score={result['final_risk_score']:.1f} "
                  f"Confidence={result['confidence']} "
                  f"ARR=${result['arr']:,.0f}")

            results.append(result)

        except Exception as e:
            print(f"    ERROR processing {account_name}: {e}")
            results.append({
                'account_id': account_id,
                'account_name': account_name,
                'arr': float(account_row['arr']),
                'plan_tier': str(account_row['plan_tier']),
                'industry': str(account_row['industry']),
                'region': str(account_row['region']),
                'csm_name': str(account_row['csm_name']),
                'contract_end_date': str(account_row['contract_end_date']),
                'days_until_renewal': int(account_row['days_until_renewal']),
                'renewal_urgency': str(account_row.get('renewal_urgency', '')),
                'quantitative_score': 0,
                'qualitative_adjustment': 0,
                'conflict_adjustment': 0,
                'final_risk_score': 50,
                'risk_tier': 'Unknown',
                'confidence': 'low',
                'usage_health': 50,
                'support_health': 50,
                'nps_health': 50,
                'csm_health': 50,
                'sdk_health': 50,
                'nps_score': None,
                'nps_category': 'unknown',
                'relationship_health': 'unknown',
                'executive_attention': False,
                'risk_factors': '[]',
                'recommended_actions': json.dumps([f'Error: {str(e)}']),
                'qualitative_factors': '[]',
                'conflict_notes': '',
                'silent_churn_patterns': '[]',
                'csm_summaries': '[]',
                'competitors': '',
            })

        time.sleep(0.5)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('final_risk_score', ascending=False)

    print("\n" + "=" * 60)
    print("RISK SCORING COMPLETE")
    print("=" * 60)

    tier_counts = results_df['risk_tier'].value_counts()
    total_arr = results_df['arr'].sum()

    for tier in ['High', 'Medium', 'Low', 'Unknown']:
        count = tier_counts.get(tier, 0)
        arr_at_risk = results_df[results_df['risk_tier'] == tier]['arr'].sum()
        print(f"  {tier}: {count} accounts | ARR: ${arr_at_risk:,.0f}")

    print(f"\nTotal Renewal ARR: ${total_arr:,.0f}")
    high_arr = results_df[results_df['risk_tier'] == 'High']['arr'].sum()
    if total_arr > 0:
        print(f"High Risk ARR: ${high_arr:,.0f} ({high_arr / total_arr * 100:.1f}%)")

    results_df.to_csv('outputs/risk_scored_accounts.csv', index=False, encoding='utf-8-sig')
    print("\nSaved: outputs/risk_scored_accounts.csv")

    return results_df