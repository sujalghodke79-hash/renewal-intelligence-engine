"""
Test Part 4 — LLM Pipeline.
Run: python test_part4.py
"""

import sys
sys.path.insert(0, '.')

import json
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.data_ingestion import load_all_data
from src.data_reconciliation import reconcile_all_data
from src.feature_engineering import build_feature_matrix
from src.llm_pipeline import run_llm_pipeline


console = Console()


def test_llm_pipeline():
    """Run full pipeline through LLM analysis."""

    console.print("\n[bold blue]Step 1: Loading data...[/bold blue]")
    data = load_all_data()

    console.print("\n[bold blue]Step 2: Reconciling data...[/bold blue]")
    reconciled = reconcile_all_data(
        accounts_df=data['accounts'],
        usage_df=data['usage_metrics'],
        support_df=data['support_tickets'],
        csm_notes_raw=data['csm_notes_raw'],
        nps_df=data['nps_responses'],
        changelog_raw=data['changelog_raw'],
        llm_provider="groq"
    )

    console.print("\n[bold blue]Step 3: Building feature matrix...[/bold blue]")
    all_features, renewal_features = build_feature_matrix(reconciled)

    console.print("\n[bold blue]Step 4: Running LLM deep analysis...[/bold blue]")
    llm_results = run_llm_pipeline(
        reconciled_data=reconciled,
        feature_matrix=all_features,
        provider="groq"
    )

    # Display Changelog Impacts
    console.print("\n")
    console.rule("[bold red]CHANGELOG IMPACT ANALYSIS[/bold red]")

    impacts = llm_results['changelog_impacts']
    impacted = impacts[impacts['changelog_impact_count'] > 0].sort_values(
        'changelog_risk_score', ascending=False
    )

    impact_table = Table(title=f"Accounts Impacted by Product Changes ({len(impacted)})")
    impact_table.add_column("ID", style="cyan")
    impact_table.add_column("Account", style="white")
    impact_table.add_column("Impacts", justify="center")
    impact_table.add_column("Critical", justify="center", style="red")
    impact_table.add_column("Risk Score", justify="center")
    impact_table.add_column("Deadline", justify="center")

    for _, row in impacted.head(15).iterrows():
        acct = all_features[all_features['account_id'] == row['account_id']]
        name = acct['account_name'].values[0] if len(acct) > 0 else '?'
        risk_color = (
            "red" if row['changelog_risk_score'] >= 50
            else "yellow" if row['changelog_risk_score'] >= 20
            else "green"
        )
        impact_table.add_row(
            str(row['account_id']),
            name[:25],
            str(row['changelog_impact_count']),
            str(row['critical_impacts']),
            f"[{risk_color}]{row['changelog_risk_score']}[/{risk_color}]",
            "YES" if row['has_deadline_pressure'] else "No"
        )

    console.print(impact_table)

    # Display Silent Churn
    console.print("\n")
    console.rule("[bold red]SILENT CHURN PATTERNS[/bold red]")

    silent = llm_results['silent_churn']
    high_risk = silent[silent['silent_churn_risk'].isin(['high', 'medium'])].sort_values(
        'silent_churn_score', ascending=False
    )

    for _, row in high_risk.iterrows():
        acct = all_features[all_features['account_id'] == row['account_id']]
        name = acct['account_name'].values[0] if len(acct) > 0 else '?'
        arr = float(acct['arr'].values[0]) if len(acct) > 0 else 0

        risk_level = row['silent_churn_risk']
        border = "red" if risk_level == 'high' else "yellow"

        pattern_text = '\n'.join([
            f"  [{p.get('confidence', '?')}] {p.get('pattern_name', '?')}: "
            f"{p.get('evidence', '?')}"
            for p in row['silent_churn_patterns']
        ]) or "  None detected"

        console.print(Panel(
            f"[bold]{name}[/bold] (ID: {row['account_id']}, ARR: ${arr:,.0f})\n"
            f"Silent Churn Risk: [bold]{risk_level.upper()}[/bold] "
            f"(Score: {row['silent_churn_score']})\n\n"
            f"Patterns Detected ({row['silent_churn_pattern_count']}):\n"
            f"{pattern_text}\n\n"
            f"Investigation: {row['recommended_investigation']}",
            title=f"Silent Churn -- {name}",
            border_style=border
        ))

    # Display Portfolio Insights
    console.print("\n")
    console.rule("[bold magenta]PORTFOLIO-LEVEL INSIGHTS[/bold magenta]")

    insights = llm_results['portfolio_insights']

    if 'portfolio_insights' in insights:
        for i, insight in enumerate(insights['portfolio_insights'], 1):
            urgency = insight.get('urgency', 'unknown')
            urgency_label = {
                'immediate': '[IMMEDIATE]',
                'this_week': '[THIS WEEK]',
                'this_month': '[THIS MONTH]',
                'this_quarter': '[THIS QUARTER]'
            }.get(urgency, '[UNKNOWN]')

            console.print(Panel(
                f"[bold]{insight.get('insight_detail', '')}[/bold]\n\n"
                f"Evidence: {insight.get('evidence', '')}\n"
                f"ARR at Risk: {insight.get('arr_at_risk', 'Unknown')}\n"
                f"Action: {insight.get('recommended_action', '')}\n"
                f"Urgency: {urgency_label}",
                title=f"Insight #{i}: {insight.get('insight_title', '')}",
                border_style="magenta"
            ))

    if 'top_priority_action' in insights:
        console.print(
            f"\n[bold red]TOP PRIORITY: {insights['top_priority_action']}[/bold red]"
        )

    console.print(
        f"\n[bold green]All outputs saved to outputs/[/bold green]"
    )

    return llm_results


if __name__ == "__main__":
    results = test_llm_pipeline()