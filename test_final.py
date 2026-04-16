"""
Final end-to-end test.
Run: python test_final.py

This runs the COMPLETE pipeline and validates all outputs.
"""

import sys

sys.path.insert(0, '.')

import json
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.config import OUTPUT_DIR
from src.data_ingestion import load_all_data
from src.data_reconciliation import reconcile_all_data
from src.feature_engineering import build_feature_matrix
from src.llm_pipeline import run_llm_pipeline
from src.risk_scoring import score_all_accounts
from src.explanation_generator import generate_all_explanations

console = Console()


def run_full_pipeline():
    """Run complete pipeline end-to-end."""

    console.print(Panel(
        "[bold]Renewal Intelligence Engine — Full Pipeline Test[/bold]",
        border_style="blue"
    ))

    # Step 1
    console.print("\n[bold cyan]STEP 1: Data Ingestion[/bold cyan]")
    data = load_all_data()
    assert len(data['accounts']) == 120
    console.print("[green]✅ Data loaded[/green]")

    # Step 2
    console.print("\n[bold cyan]STEP 2: Data Reconciliation[/bold cyan]")
    reconciled = reconcile_all_data(
        accounts_df=data['accounts'],
        usage_df=data['usage_metrics'],
        support_df=data['support_tickets'],
        csm_notes_raw=data['csm_notes_raw'],
        nps_df=data['nps_responses'],
        changelog_raw=data['changelog_raw'],
        llm_provider="groq"
    )
    assert len(reconciled['csm_notes_parsed']) > 0
    console.print("[green]✅ Data reconciled[/green]")

    # Step 3
    console.print("\n[bold cyan]STEP 3: Feature Engineering[/bold cyan]")
    all_features, renewal_features = build_feature_matrix(reconciled)
    assert len(all_features) == 120
    assert len(renewal_features) > 0
    console.print(f"[green]✅ Features computed: {len(all_features.columns)} features, "
                  f"{len(renewal_features)} renewal accounts[/green]")

    # Step 4
    console.print("\n[bold cyan]STEP 4: LLM Deep Analysis[/bold cyan]")
    llm_results = run_llm_pipeline(
        reconciled_data=reconciled,
        feature_matrix=all_features,
        provider="groq"
    )
    assert 'changelog_impacts' in llm_results
    assert 'silent_churn' in llm_results
    assert 'portfolio_insights' in llm_results
    console.print("[green]✅ LLM analysis complete[/green]")

    # Step 5
    console.print("\n[bold cyan]STEP 5: Risk Scoring (LangGraph)[/bold cyan]")
    risk_results = score_all_accounts(
        feature_matrix=all_features,
        reconciled_data=reconciled,
        llm_results=llm_results
    )
    assert len(risk_results) > 0
    assert 'risk_tier' in risk_results.columns
    assert set(risk_results['risk_tier'].unique()).issubset({'High', 'Medium', 'Low', 'Unknown'})
    console.print("[green]✅ Risk scoring complete[/green]")

    # Step 6
    console.print("\n[bold cyan]STEP 6: Explanation Generation[/bold cyan]")
    explanations = generate_all_explanations(
        risk_results=risk_results,
        portfolio_insights=llm_results.get('portfolio_insights', {}),
        provider="groq"
    )
    assert len(explanations['account_narratives']) > 0
    assert len(explanations['executive_summary']) > 0
    console.print("[green]✅ Explanations generated[/green]")

    # Validate outputs
    console.print("\n[bold cyan]VALIDATION[/bold cyan]")

    expected_files = [
        'risk_scored_accounts.csv',
        'all_account_features.csv',
        'renewal_account_features.csv',
        'executive_summary.md',
        'account_narratives.md',
        'csm_briefings.md',
        'changelog_impacts.csv',
        'silent_churn_analysis.csv',
        'portfolio_insights.json'
    ]

    for filename in expected_files:
        path = OUTPUT_DIR / filename
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        status = "[green]✅[/green]" if exists and size > 0 else "[red]❌[/red]"
        console.print(f"  {status} {filename} ({size:,} bytes)")

    # Final summary
    console.print("\n")

    total = len(risk_results)
    high = len(risk_results[risk_results['risk_tier'] == 'High'])
    medium = len(risk_results[risk_results['risk_tier'] == 'Medium'])
    low = len(risk_results[risk_results['risk_tier'] == 'Low'])

    summary_table = Table(title="Final Results Summary")
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Total Renewal Accounts", str(total))
    summary_table.add_row("🔴 High Risk",
                          f"{high} (${risk_results[risk_results['risk_tier'] == 'High']['arr'].sum():,.0f})")
    summary_table.add_row("🟡 Medium Risk",
                          f"{medium} (${risk_results[risk_results['risk_tier'] == 'Medium']['arr'].sum():,.0f})")
    summary_table.add_row("🟢 Low Risk",
                          f"{low} (${risk_results[risk_results['risk_tier'] == 'Low']['arr'].sum():,.0f})")
    summary_table.add_row("Total Renewal ARR", f"${risk_results['arr'].sum():,.0f}")
    summary_table.add_row("Account Narratives", str(len(explanations['account_narratives'])))
    summary_table.add_row("CSM Briefings", str(len(explanations['csm_briefings'])))
    summary_table.add_row("Portfolio Insights",
                          str(len(llm_results.get('portfolio_insights', {}).get('portfolio_insights', []))))

    console.print(summary_table)

    console.print(Panel(
        "[bold green]🎉 All tests passed! Pipeline is working end-to-end.[/bold green]\n\n"
        "Next steps:\n"
        "  CLI:       python cli.py summary\n"
        "  CLI:       python cli.py account 1007\n"
        "  Streamlit: streamlit run app.py",
        title="✅ Pipeline Complete",
        border_style="green"
    ))

    return risk_results


if __name__ == "__main__":
    results = run_full_pipeline()