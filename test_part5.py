"""
Test Part 5 — Risk Scoring with LangGraph.
Run: python test_part5.py

Full pipeline: ingestion → reconciliation → features → LLM analysis → risk scoring
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
from src.risk_scoring import score_all_accounts

console = Console()


def test_risk_scoring():
    """Run complete pipeline through risk scoring."""

    # Step 1-3: Load → Reconcile → Features
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

    console.print("\n[bold blue]Step 3: Building features...[/bold blue]")
    all_features, renewal_features = build_feature_matrix(reconciled)

    # Step 4: LLM Pipeline
    console.print("\n[bold blue]Step 4: Running LLM analysis...[/bold blue]")
    llm_results = run_llm_pipeline(
        reconciled_data=reconciled,
        feature_matrix=all_features,
        provider="groq"
    )

    # Step 5: Risk Scoring
    console.print("\n[bold blue]Step 5: Scoring accounts with LangGraph...[/bold blue]")
    risk_results = score_all_accounts(
        feature_matrix=all_features,
        reconciled_data=reconciled,
        llm_results=llm_results
    )

    # ==============================
    # Display Final Results
    # ==============================

    console.print("\n")
    console.rule("[bold red]FINAL RISK ASSESSMENT — RENEWAL ACCOUNTS[/bold red]")

    # Risk tier summary
    summary_table = Table(title="Risk Tier Summary")
    summary_table.add_column("Tier", style="bold")
    summary_table.add_column("Count", justify="center")
    summary_table.add_column("ARR at Risk", justify="right", style="green")
    summary_table.add_column("% of Total ARR", justify="right")

    total_arr = risk_results['arr'].sum()
    for tier, emoji, color in [('High', '🔴', 'red'), ('Medium', '🟡', 'yellow'), ('Low', '🟢', 'green')]:
        tier_data = risk_results[risk_results['risk_tier'] == tier]
        tier_arr = tier_data['arr'].sum()
        pct = (tier_arr / total_arr * 100) if total_arr > 0 else 0
        summary_table.add_row(
            f"{emoji} {tier}",
            str(len(tier_data)),
            f"${tier_arr:,.0f}",
            f"{pct:.1f}%"
        )

    console.print(summary_table)

    # Detailed table — High Risk
    console.print("\n")
    high_risk = risk_results[risk_results['risk_tier'] == 'High'].sort_values(
        'final_risk_score', ascending=False
    )

    if len(high_risk) > 0:
        high_table = Table(title=f"🔴 HIGH RISK ACCOUNTS ({len(high_risk)})")
        high_table.add_column("ID", style="cyan")
        high_table.add_column("Account", style="white")
        high_table.add_column("ARR", justify="right", style="green")
        high_table.add_column("Score", justify="center")
        high_table.add_column("Days", justify="center")
        high_table.add_column("Confidence", justify="center")
        high_table.add_column("Top Risk Factor", style="red")

        for _, row in high_risk.iterrows():
            # Parse risk factors
            try:
                factors = json.loads(row['risk_factors']) if isinstance(row['risk_factors'], str) else row[
                    'risk_factors']
                top_factor = factors[0]['factor'] if factors else 'N/A'
            except:
                top_factor = 'N/A'

            high_table.add_row(
                str(row['account_id']),
                str(row['account_name'])[:25],
                f"${row['arr']:,.0f}",
                f"[red]{row['final_risk_score']:.0f}[/red]",
                str(int(row['days_until_renewal'])),
                row['confidence'],
                top_factor
            )

        console.print(high_table)

    # Detailed table — Medium Risk
    medium_risk = risk_results[risk_results['risk_tier'] == 'Medium'].sort_values(
        'final_risk_score', ascending=False
    )

    if len(medium_risk) > 0:
        med_table = Table(title=f"🟡 MEDIUM RISK ACCOUNTS ({len(medium_risk)})")
        med_table.add_column("ID", style="cyan")
        med_table.add_column("Account", style="white")
        med_table.add_column("ARR", justify="right", style="green")
        med_table.add_column("Score", justify="center")
        med_table.add_column("Days", justify="center")
        med_table.add_column("Confidence", justify="center")

        for _, row in medium_risk.iterrows():
            med_table.add_row(
                str(row['account_id']),
                str(row['account_name'])[:25],
                f"${row['arr']:,.0f}",
                f"[yellow]{row['final_risk_score']:.0f}[/yellow]",
                str(int(row['days_until_renewal'])),
                row['confidence']
            )

        console.print(med_table)

    # Show detailed view of top 3 high risk accounts
    console.print("\n")
    console.rule("[bold]DETAILED HIGH RISK ACCOUNT PROFILES[/bold]")

    for _, row in high_risk.head(3).iterrows():
        try:
            actions = json.loads(row['recommended_actions']) if isinstance(row['recommended_actions'], str) else row[
                'recommended_actions']
            factors = json.loads(row['risk_factors']) if isinstance(row['risk_factors'], str) else row['risk_factors']
            qual_factors = json.loads(row['qualitative_factors']) if isinstance(row['qualitative_factors'], str) else \
            row['qualitative_factors']
        except:
            actions, factors, qual_factors = [], [], []

        detail_text = (
                f"[bold]ARR:[/bold] ${row['arr']:,.0f} | "
                f"[bold]Plan:[/bold] {row['plan_tier']} | "
                f"[bold]Renewal:[/bold] {row['days_until_renewal']:.0f} days | "
                f"[bold]CSM:[/bold] {row['csm_name']}\n\n"

                f"[bold]Risk Score:[/bold] [red]{row['final_risk_score']:.0f}/100[/red] "
                f"(Quant: {row['quantitative_score']:.0f} + "
                f"Qual: {row['qualitative_adjustment']:+.0f} + "
                f"Conflict: {row['conflict_adjustment']:+.0f})\n"
                f"[bold]Confidence:[/bold] {row['confidence']} | "
                f"[bold]Relationship:[/bold] {row['relationship_health']}\n\n"

                f"[bold]Health Scores:[/bold] "
                f"Usage={row['usage_health']:.0f} | "
                f"Support={row['support_health']:.0f} | "
                f"NPS={row['nps_health']:.0f} | "
                f"CSM={row['csm_health']:.0f} | "
                f"SDK={row['sdk_health']:.0f}\n\n"

                f"[bold]Competitors:[/bold] {row['competitors'] or 'None mentioned'}\n\n"

                f"[bold]Risk Factors:[/bold]\n" +
                '\n'.join([f"  • [{f.get('severity', '?')}] {f.get('factor', '?')}: {f.get('detail', '')}" for f in
                           factors[:5]]) +
                '\n\n'

                f"[bold]Recommended Actions:[/bold]\n" +
                '\n'.join([f"  {a}" for a in actions])
        )

        console.print(Panel(
            detail_text,
            title=f"🔴 {row['account_name']} (ID: {row['account_id']})",
            border_style="red",
            width=100
        ))

    console.print(
        f"\n[bold green]✅ Risk scoring complete! Results saved to outputs/risk_scored_accounts.csv[/bold green]")

    return risk_results


if __name__ == "__main__":
    results = test_risk_scoring()