"""
Explanation Generator Module

Generates plain-English explanations for each at-risk account:
1. Per-account risk narrative
2. Executive summary for the portfolio
3. CSM-specific briefings
4. Non-obvious insight highlights

Uses LangChain for narrative generation.
"""

import json
import time
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import date

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

from src.config import REFERENCE_DATE, RENEWAL_CUTOFF_DATE
from src.utils import get_llm


# ============================================================
# 1. PER-ACCOUNT RISK NARRATIVE
# ============================================================

def generate_account_narrative(
    account_row: pd.Series,
    provider: str = "groq"
) -> str:
    """
    Generate a plain-English risk narrative for a single account.
    Written as if briefing a VP of Customer Success.
    """
    llm = get_llm(provider=provider, temperature=0.3, max_tokens=2000)

    def safe_parse(val, default=None):
        if default is None:
            default = []
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return json.loads(val)
            except:
                return default
        return default

    risk_factors = safe_parse(account_row.get('risk_factors'))
    actions = safe_parse(account_row.get('recommended_actions'))
    qual_factors = safe_parse(account_row.get('qualitative_factors'))
    silent_patterns = safe_parse(account_row.get('silent_churn_patterns'))
    csm_summaries = safe_parse(account_row.get('csm_summaries'))

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior Customer Success strategist writing an internal risk brief.

Write in a professional but direct style. No fluff. The audience is the VP of CS and the account's CSM.

Structure your narrative as:
1. **Risk Summary** (2-3 sentences: what's happening, how urgent)
2. **Key Signals** (bullet points of the most important signals, mixing quantitative and qualitative)
3. **What We Might Be Missing** (any data gaps, contradictions, or uncertain signals)
4. **Recommended Next Steps** (specific, time-bound actions)

Use plain English. Avoid jargon. Quantify where possible.
Do NOT use generic phrases like "proactive engagement" — be specific about what to DO.
Do NOT use emoji characters in your response."""),
        ("human", """Write a risk narrative for this account:

ACCOUNT: {account_name}
- Account ID: {account_id}
- ARR: ${arr}
- Plan: {plan_tier} | Industry: {industry} | Region: {region}
- CSM: {csm_name}
- Contract ends: {contract_end} ({days_left} days from now)
- Risk Tier: {risk_tier} | Risk Score: {risk_score}/100
- Confidence: {confidence}

HEALTH SCORES (0=critical, 100=healthy):
- Usage: {usage_health}/100
- Support: {support_health}/100
- NPS: {nps_health}/100 (Score: {nps_score}, Category: {nps_category})
- CSM Sentiment: {csm_health}/100
- SDK/Platform: {sdk_health}/100
- Relationship: {relationship_health}

RISK FACTORS:
{risk_factors_text}

QUALITATIVE FACTORS:
{qual_factors_text}

CSM NOTES SUMMARY:
{csm_notes_text}

SILENT CHURN PATTERNS:
{silent_patterns_text}

COMPETITORS MENTIONED: {competitors}

CONFLICT NOTES: {conflict_notes}

RECOMMENDED ACTIONS:
{actions_text}""")
    ])

    chain = prompt | llm | StrOutputParser()

    risk_factors_text = '\n'.join([
        f"- [{f.get('severity', '?')}] {f.get('factor', '?')}: {f.get('detail', '')}"
        for f in risk_factors
    ]) or "No significant risk factors identified"

    qual_factors_text = '\n'.join([
        f"- {f}" for f in qual_factors
    ]) or "No qualitative factors noted"

    csm_notes_text = '\n'.join([
        f"- {s}" for s in csm_summaries
    ]) or "No CSM notes available for this account"

    silent_patterns_text = '\n'.join([
        f"- [{p.get('confidence', '?')}] {p.get('pattern_name', '?')}: {p.get('evidence', '')}"
        for p in silent_patterns
    ]) or "No silent churn patterns detected"

    actions_text = '\n'.join([
        f"- {a}" for a in actions
    ]) or "Standard renewal process"

    try:
        narrative = chain.invoke({
            "account_name": account_row.get('account_name', ''),
            "account_id": account_row.get('account_id', ''),
            "arr": f"{account_row.get('arr', 0):,.0f}",
            "plan_tier": account_row.get('plan_tier', ''),
            "industry": account_row.get('industry', ''),
            "region": account_row.get('region', ''),
            "csm_name": account_row.get('csm_name', ''),
            "contract_end": account_row.get('contract_end_date', ''),
            "days_left": int(account_row.get('days_until_renewal', 0)),
            "risk_tier": account_row.get('risk_tier', ''),
            "risk_score": account_row.get('final_risk_score', 0),
            "confidence": account_row.get('confidence', ''),
            "usage_health": account_row.get('usage_health', 50),
            "support_health": account_row.get('support_health', 50),
            "nps_health": account_row.get('nps_health', 50),
            "nps_score": account_row.get('nps_score', 'N/A') if pd.notna(account_row.get('nps_score')) else 'N/A',
            "nps_category": account_row.get('nps_category', 'unknown'),
            "csm_health": account_row.get('csm_health', 50),
            "sdk_health": account_row.get('sdk_health', 50),
            "relationship_health": account_row.get('relationship_health', 'unknown'),
            "risk_factors_text": risk_factors_text,
            "qual_factors_text": qual_factors_text,
            "csm_notes_text": csm_notes_text,
            "silent_patterns_text": silent_patterns_text,
            "competitors": account_row.get('competitors', 'None'),
            "conflict_notes": account_row.get('conflict_notes', 'None'),
            "actions_text": actions_text
        })

        return narrative

    except Exception as e:
        return f"Error generating narrative: {str(e)}"


# ============================================================
# 2. EXECUTIVE SUMMARY
# ============================================================

def generate_executive_summary(
    risk_results: pd.DataFrame,
    portfolio_insights: Dict[str, Any],
    provider: str = "groq"
) -> str:
    """
    Generate a portfolio-level executive summary.
    """
    llm = get_llm(provider=provider, temperature=0.3, max_tokens=3000)

    total_accounts = len(risk_results)
    total_arr = risk_results['arr'].sum()

    high_risk = risk_results[risk_results['risk_tier'] == 'High']
    medium_risk = risk_results[risk_results['risk_tier'] == 'Medium']
    low_risk = risk_results[risk_results['risk_tier'] == 'Low']

    high_arr = high_risk['arr'].sum()
    medium_arr = medium_risk['arr'].sum()

    csm_risk = risk_results.groupby('csm_name').agg({
        'account_id': 'count',
        'arr': 'sum',
        'final_risk_score': 'mean'
    }).rename(columns={
        'account_id': 'accounts',
        'arr': 'total_arr',
        'final_risk_score': 'avg_risk'
    }).sort_values('avg_risk', ascending=False)

    csm_summary = '\n'.join([
        f"  - {name}: {row['accounts']} accounts, "
        f"${row['total_arr']:,.0f} ARR, "
        f"avg risk: {row['avg_risk']:.0f}/100"
        for name, row in csm_risk.iterrows()
    ])

    top_risk = high_risk.head(5)
    top_risk_text = '\n'.join([
        f"  {i+1}. {row['account_name']} -- ${row['arr']:,.0f} ARR, "
        f"Score: {row['final_risk_score']:.0f}/100, "
        f"Renews in {int(row['days_until_renewal'])} days"
        for i, (_, row) in enumerate(top_risk.iterrows())
    ])

    insights_text = '\n'.join([
        f"  - {ins.get('insight_title', '?')}: {ins.get('insight_detail', '')}"
        for ins in portfolio_insights.get('portfolio_insights', [])
    ]) or "No portfolio-level insights generated"

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a BizOps analyst writing a quarterly renewal risk briefing for the CRO.

Write a concise, data-driven executive summary. Use specific numbers. 
Highlight what needs attention THIS WEEK vs THIS MONTH.
Keep it to approximately 500 words. Use markdown formatting.
Do NOT use emoji characters. Use text labels like [HIGH RISK] instead."""),
        ("human", """Write the executive summary for this quarter's renewal risk report.

PORTFOLIO OVERVIEW:
- Renewal accounts: {total_accounts}
- Total renewal ARR: ${total_arr}
- High risk: {high_count} accounts, ${high_arr} ARR ({high_pct:.1f}%)
- Medium risk: {medium_count} accounts, ${medium_arr} ARR ({medium_pct:.1f}%)
- Low risk: {low_count} accounts, ${low_arr} ARR ({low_pct:.1f}%)
- Reference date: {ref_date}
- Renewal window: next 90 days (through {cutoff_date})

TOP 5 AT-RISK ACCOUNTS:
{top_risk_text}

CSM WORKLOAD:
{csm_summary}

PORTFOLIO-LEVEL INSIGHTS:
{insights_text}

TOP PRIORITY ACTION: {top_priority}""")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        summary = chain.invoke({
            "total_accounts": total_accounts,
            "total_arr": f"{total_arr:,.0f}",
            "high_count": len(high_risk),
            "high_arr": f"{high_arr:,.0f}",
            "high_pct": (high_arr / total_arr * 100) if total_arr > 0 else 0,
            "medium_count": len(medium_risk),
            "medium_arr": f"{medium_arr:,.0f}",
            "medium_pct": (medium_arr / total_arr * 100) if total_arr > 0 else 0,
            "low_count": len(low_risk),
            "low_arr": f"{low_risk['arr'].sum():,.0f}",
            "low_pct": (low_risk['arr'].sum() / total_arr * 100) if total_arr > 0 else 0,
            "ref_date": REFERENCE_DATE.isoformat(),
            "cutoff_date": RENEWAL_CUTOFF_DATE.isoformat(),
            "top_risk_text": top_risk_text or "No high risk accounts identified",
            "csm_summary": csm_summary,
            "insights_text": insights_text,
            "top_priority": portfolio_insights.get('top_priority_action', 'N/A')
        })

        return summary

    except Exception as e:
        return f"Error generating executive summary: {str(e)}"


# ============================================================
# 3. CSM BRIEFING
# ============================================================

def generate_csm_briefing(
    risk_results: pd.DataFrame,
    csm_name: str,
    provider: str = "groq"
) -> str:
    """
    Generate a briefing for a specific CSM about their at-risk accounts.
    """
    llm = get_llm(provider=provider, temperature=0.3, max_tokens=2000)

    csm_accounts = risk_results[risk_results['csm_name'] == csm_name].sort_values(
        'final_risk_score', ascending=False
    )

    if len(csm_accounts) == 0:
        return f"No renewal accounts assigned to {csm_name} in the current window."

    accounts_text = '\n'.join([
        f"- {row['account_name']} (${row['arr']:,.0f} ARR) -- "
        f"Risk: {row['risk_tier']} ({row['final_risk_score']:.0f}/100), "
        f"Renews in {int(row['days_until_renewal'])} days, "
        f"Competitors: {row.get('competitors', 'None') or 'None'}"
        for _, row in csm_accounts.iterrows()
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are writing a weekly briefing for a Customer Success Manager.
Be practical and specific. Focus on what they should DO this week.
Use markdown. Keep it concise. Do NOT use emoji characters."""),
        ("human", """Write a renewal briefing for CSM: {csm_name}

Their renewal accounts ({count} total, ${total_arr} ARR):
{accounts_text}

High risk accounts: {high_count}
Medium risk accounts: {medium_count}

Write:
1. This week's priorities (top 2-3 actions)
2. Account-by-account status (one line each)
3. Escalation needs (accounts requiring leadership involvement)""")
    ])

    chain = prompt | llm | StrOutputParser()

    try:
        briefing = chain.invoke({
            "csm_name": csm_name,
            "count": len(csm_accounts),
            "total_arr": f"{csm_accounts['arr'].sum():,.0f}",
            "accounts_text": accounts_text,
            "high_count": len(csm_accounts[csm_accounts['risk_tier'] == 'High']),
            "medium_count": len(csm_accounts[csm_accounts['risk_tier'] == 'Medium'])
        })
        return briefing
    except Exception as e:
        return f"Error generating briefing: {str(e)}"


# ============================================================
# HELPER: Safe file write with UTF-8
# ============================================================

def safe_write_file(filepath: str, content: str):
    """Write content to file with UTF-8 encoding (Windows-safe)."""
    with open(filepath, 'w', encoding='utf-8', errors='replace') as f:
        f.write(content)


# ============================================================
# 4. MASTER EXPLANATION PIPELINE
# ============================================================

def generate_all_explanations(
    risk_results: pd.DataFrame,
    portfolio_insights: Dict[str, Any],
    provider: str = "groq"
) -> Dict[str, Any]:
    """
    Generate all explanations and narratives.
    """
    print("\n" + "=" * 60)
    print("EXPLANATION GENERATION")
    print("=" * 60)

    # 1. Account narratives (High and Medium risk only)
    print("\n[Explanations] Generating account narratives...")
    at_risk = risk_results[risk_results['risk_tier'].isin(['High', 'Medium'])].sort_values(
        'final_risk_score', ascending=False
    )

    narratives = {}
    for idx, (_, row) in enumerate(at_risk.iterrows()):
        account_id = row['account_id']
        account_name = row['account_name']
        print(f"  [{idx+1}/{len(at_risk)}] {account_name}...")

        narrative = generate_account_narrative(row, provider=provider)
        narratives[account_id] = {
            'account_name': account_name,
            'risk_tier': row['risk_tier'],
            'risk_score': row['final_risk_score'],
            'arr': row['arr'],
            'narrative': narrative
        }

        time.sleep(0.5)

    # 2. Executive summary
    print("\n[Explanations] Generating executive summary...")
    exec_summary = generate_executive_summary(
        risk_results, portfolio_insights, provider=provider
    )

    # 3. CSM briefings
    print("\n[Explanations] Generating CSM briefings...")
    csm_names = risk_results['csm_name'].unique()
    csm_briefings = {}

    for csm in csm_names:
        csm_accounts = risk_results[risk_results['csm_name'] == csm]
        has_risk = csm_accounts['risk_tier'].isin(['High', 'Medium']).any()

        if has_risk:
            print(f"  Generating briefing for {csm}...")
            briefing = generate_csm_briefing(risk_results, csm, provider=provider)
            csm_briefings[csm] = briefing
            time.sleep(0.5)

    explanations = {
        'account_narratives': narratives,
        'executive_summary': exec_summary,
        'csm_briefings': csm_briefings
    }

    # Save outputs — ALL with UTF-8 encoding
    print("\n[Explanations] Saving outputs...")

    # Save executive summary
    safe_write_file('outputs/executive_summary.md', exec_summary)

    # Save account narratives
    narratives_content = "# Account Risk Narratives\n\n"
    for aid, data in narratives.items():
        # Use text labels instead of emoji for file safety
        tier_label = {
            'High': '[HIGH RISK]',
            'Medium': '[MEDIUM RISK]',
            'Low': '[LOW RISK]'
        }.get(data['risk_tier'], '[UNKNOWN]')

        narratives_content += f"## {tier_label} {data['account_name']} (ID: {aid})\n"
        narratives_content += f"**ARR:** ${data['arr']:,.0f} | "
        narratives_content += f"**Risk:** {data['risk_tier']} ({data['risk_score']:.0f}/100)\n\n"
        narratives_content += data['narrative']
        narratives_content += "\n\n---\n\n"

    safe_write_file('outputs/account_narratives.md', narratives_content)

    # Save CSM briefings
    briefings_content = "# CSM Renewal Briefings\n\n"
    for csm, briefing in csm_briefings.items():
        briefings_content += f"## {csm}\n\n"
        briefings_content += briefing
        briefings_content += "\n\n---\n\n"

    safe_write_file('outputs/csm_briefings.md', briefings_content)

    print(f"\n  Saved: executive_summary.md, account_narratives.md, csm_briefings.md")

    print("\n" + "=" * 60)
    print("EXPLANATION GENERATION COMPLETE")
    print(f"  Account narratives: {len(narratives)}")
    print(f"  CSM briefings: {len(csm_briefings)}")
    print("=" * 60)

    return explanations