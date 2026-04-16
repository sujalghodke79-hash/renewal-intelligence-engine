"""
Renewal Intelligence Engine -- CLI Interface

Usage:
    python cli.py run
    python cli.py summary
    python cli.py account 1007
    python cli.py csm "Sarah Chen"
    python cli.py list
"""

import sys
sys.path.insert(0, '.')

import json
import argparse
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

from src.config import OUTPUT_DIR, REFERENCE_DATE, RENEWAL_CUTOFF_DATE

console = Console()


# ============================================================
# HELPERS
# ============================================================

def safe_parse_json(val):
    """Parse JSON safely from string or list."""
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def read_file(path: Path) -> str:
    """Read file with UTF-8 encoding."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def load_risk_results() -> pd.DataFrame:
    """Load risk scored accounts CSV."""
    csv_path = OUTPUT_DIR / 'risk_scored_accounts.csv'
    if not csv_path.exists():
        return None
    return pd.read_csv(csv_path, encoding='utf-8-sig')


# ============================================================
# COMMANDS
# ============================================================

def run_full_pipeline(provider: str = "groq"):
    """Run the complete pipeline."""
    from src.data_ingestion import load_all_data
    from src.data_reconciliation import reconcile_all_data
    from src.feature_engineering import build_feature_matrix
    from src.llm_pipeline import run_llm_pipeline
    from src.risk_scoring import score_all_accounts
    from src.explanation_generator import generate_all_explanations

    console.print(Panel(
        f"[bold]Renewal Intelligence Engine[/bold]\n"
        f"Reference Date: {REFERENCE_DATE}\n"
        f"Renewal Window: {REFERENCE_DATE} to {RENEWAL_CUTOFF_DATE}\n"
        f"LLM Provider: {provider}",
        title="Starting Full Pipeline",
        border_style="blue"
    ))

    with console.status("[bold blue]Step 1/6: Loading data..."):
        data = load_all_data()
    console.print("[green]Step 1: Data loaded[/green]")

    console.print("[bold blue]Step 2/6: Reconciling data (LLM calls)...[/bold blue]")
    reconciled = reconcile_all_data(
        accounts_df=data['accounts'],
        usage_df=data['usage_metrics'],
        support_df=data['support_tickets'],
        csm_notes_raw=data['csm_notes_raw'],
        nps_df=data['nps_responses'],
        changelog_raw=data['changelog_raw'],
        llm_provider=provider
    )
    console.print("[green]Step 2: Data reconciled[/green]")

    with console.status("[bold blue]Step 3/6: Computing features..."):
        all_features, renewal_features = build_feature_matrix(reconciled)
    console.print("[green]Step 3: Features computed[/green]")

    console.print("[bold blue]Step 4/6: LLM deep analysis...[/bold blue]")
    llm_results = run_llm_pipeline(
        reconciled_data=reconciled,
        feature_matrix=all_features,
        provider=provider
    )
    console.print("[green]Step 4: LLM analysis complete[/green]")

    console.print("[bold blue]Step 5/6: Scoring with LangGraph...[/bold blue]")
    risk_results = score_all_accounts(
        feature_matrix=all_features,
        reconciled_data=reconciled,
        llm_results=llm_results
    )
    console.print("[green]Step 5: Risk scoring complete[/green]")

    console.print("[bold blue]Step 6/6: Generating explanations...[/bold blue]")
    explanations = generate_all_explanations(
        risk_results=risk_results,
        portfolio_insights=llm_results.get('portfolio_insights', {}),
        provider=provider
    )
    console.print("[green]Step 6: Explanations generated[/green]")

    display_summary(risk_results)

    console.print(Panel(
        "[bold green]Pipeline complete![/bold green]\n\n"
        "Output files in outputs/:\n"
        "  risk_scored_accounts.csv\n"
        "  executive_summary.md\n"
        "  account_narratives.md\n"
        "  csm_briefings.md\n"
        "  changelog_impacts.csv\n"
        "  silent_churn_analysis.csv\n"
        "  portfolio_insights.json",
        title="Complete",
        border_style="green"
    ))


def display_summary(risk_results: pd.DataFrame = None):
    """Display executive summary."""
    if risk_results is None:
        risk_results = load_risk_results()
        if risk_results is None:
            console.print(
                "[red]No results found. Run 'python cli.py run' first.[/red]"
            )
            return

    total_arr = risk_results['arr'].sum()

    summary_table = Table(title="Renewal Risk Summary")
    summary_table.add_column("Risk Tier", style="bold")
    summary_table.add_column("Accounts", justify="center")
    summary_table.add_column("ARR", justify="right")
    summary_table.add_column("% of ARR", justify="right")
    summary_table.add_column("Avg Score", justify="center")

    for tier, color in [('High', 'red'), ('Medium', 'yellow'), ('Low', 'green')]:
        tier_data = risk_results[risk_results['risk_tier'] == tier]
        tier_arr = tier_data['arr'].sum()
        pct = (tier_arr / total_arr * 100) if total_arr > 0 else 0
        avg_score = tier_data['final_risk_score'].mean() if len(tier_data) > 0 else 0

        summary_table.add_row(
            f"[{color}]{tier}[/{color}]",
            str(len(tier_data)),
            f"[{color}]${tier_arr:,.0f}[/{color}]",
            f"{pct:.1f}%",
            f"{avg_score:.0f}"
        )

    summary_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{len(risk_results)}[/bold]",
        f"[bold]${total_arr:,.0f}[/bold]",
        "100%",
        f"{risk_results['final_risk_score'].mean():.0f}"
    )

    console.print(summary_table)

    # High risk table
    high_risk = risk_results[risk_results['risk_tier'] == 'High'].sort_values(
        'final_risk_score', ascending=False
    )

    if len(high_risk) > 0:
        risk_table = Table(title="High Risk Accounts -- Immediate Attention Required")
        risk_table.add_column("ID", style="cyan", width=6)
        risk_table.add_column("Account", width=25)
        risk_table.add_column("ARR", justify="right", width=12)
        risk_table.add_column("Score", justify="center", width=7)
        risk_table.add_column("Days", justify="center", width=6)
        risk_table.add_column("CSM", width=15)
        risk_table.add_column("Confidence", width=10)
        risk_table.add_column("Competitors", width=20)

        for _, row in high_risk.iterrows():
            risk_table.add_row(
                str(int(row['account_id'])),
                str(row['account_name'])[:25],
                f"${row['arr']:,.0f}",
                f"[red]{row['final_risk_score']:.0f}[/red]",
                str(int(row['days_until_renewal'])),
                str(row['csm_name']),
                str(row['confidence']),
                str(row.get('competitors', '') or '')[:20]
            )

        console.print(risk_table)

    # Executive summary
    exec_path = OUTPUT_DIR / 'executive_summary.md'
    if exec_path.exists():
        console.rule("[bold]Executive Summary[/bold]")
        console.print(Markdown(read_file(exec_path)))


def show_account_detail(account_id: int):
    """Show detailed risk profile for an account."""
    risk_results = load_risk_results()
    if risk_results is None:
        console.print("[red]No results found. Run 'python cli.py run' first.[/red]")
        return

    account = risk_results[risk_results['account_id'] == account_id]

    if len(account) == 0:
        console.print(f"[red]Account {account_id} not in renewal window.[/red]")
        all_path = OUTPUT_DIR / 'all_account_features.csv'
        if all_path.exists():
            all_accounts = pd.read_csv(all_path, encoding='utf-8-sig')
            match = all_accounts[all_accounts['account_id'] == account_id]
            if len(match) > 0:
                end = match.iloc[0].get('contract_end_date', '?')
                console.print(
                    f"[yellow]Account exists but renews {end} "
                    f"(outside 90-day window).[/yellow]"
                )
        return

    row = account.iloc[0]
    risk_factors = safe_parse_json(row.get('risk_factors'))
    actions = safe_parse_json(row.get('recommended_actions'))

    tier = row['risk_tier']
    tier_label = {'High': '[HIGH RISK]', 'Medium': '[MEDIUM RISK]', 'Low': '[LOW RISK]'}.get(
        tier, '[UNKNOWN]'
    )
    border = {'High': 'red', 'Medium': 'yellow', 'Low': 'green'}.get(tier, 'white')

    # Build health bars
    def health_bar(val):
        val = float(val) if val is not None else 50
        filled = int(val / 5)
        empty = 20 - filled
        return f"{'|' * filled}{'.' * empty} {val:.0f}"

    detail = (
        f"[bold]Account:[/bold] {row['account_name']} (ID: {int(row['account_id'])})\n"
        f"[bold]ARR:[/bold] ${row['arr']:,.0f} | "
        f"[bold]Plan:[/bold] {row['plan_tier']} | "
        f"[bold]Industry:[/bold] {row['industry']} | "
        f"[bold]Region:[/bold] {row['region']}\n"
        f"[bold]CSM:[/bold] {row['csm_name']}\n"
        f"[bold]Contract End:[/bold] {row['contract_end_date']} "
        f"({int(row['days_until_renewal'])} days)\n\n"
        f"[bold]Risk: {tier_label} {row['final_risk_score']:.0f}/100[/bold]\n"
        f"Quant={row['quantitative_score']:.0f} + "
        f"Qual={row['qualitative_adjustment']:+.0f} + "
        f"Conflict={row['conflict_adjustment']:+.0f}\n"
        f"Confidence: {row['confidence']} | "
        f"Relationship: {row.get('relationship_health', 'unknown')}\n\n"
        f"[bold]Health Scores:[/bold]\n"
        f"  Usage:   {health_bar(row['usage_health'])}\n"
        f"  Support: {health_bar(row['support_health'])}\n"
        f"  NPS:     {health_bar(row['nps_health'])}\n"
        f"  CSM:     {health_bar(row['csm_health'])}\n"
        f"  SDK:     {health_bar(row['sdk_health'])}\n\n"
        f"[bold]Competitors:[/bold] {row.get('competitors', '') or 'None'}\n"
        f"[bold]Exec Attention:[/bold] "
        f"{'Yes' if row.get('executive_attention') else 'No'}\n\n"
        f"[bold]Risk Factors:[/bold]\n" +
        '\n'.join([
            f"  [{f.get('severity', '?')}] {f.get('factor', '?')}: {f.get('detail', '')}"
            for f in risk_factors[:6]
        ]) +
        f"\n\n[bold]Recommended Actions:[/bold]\n" +
        '\n'.join([f"  {a}" for a in actions])
    )

    console.print(Panel(
        detail,
        title=f"Account Detail -- {row['account_name']}",
        border_style=border,
        width=100
    ))

    # Show narrative
    narrative_path = OUTPUT_DIR / 'account_narratives.md'
    if narrative_path.exists():
        content = read_file(narrative_path)
        account_name = str(row['account_name'])

        sections = content.split("---")
        for section in sections:
            if account_name in section:
                console.rule("[bold]Risk Narrative[/bold]")
                console.print(Markdown(section.strip()))
                break


def show_csm_briefing(csm_name: str):
    """Show CSM briefing."""
    briefing_path = OUTPUT_DIR / 'csm_briefings.md'

    if not briefing_path.exists():
        console.print("[red]No briefings found. Run 'python cli.py run' first.[/red]")
        return

    content = read_file(briefing_path)
    marker = f"## {csm_name}"

    if marker in content:
        start = content.index(marker)
        end = content.find("\n---\n", start)
        if end == -1:
            end = len(content)
        section = content[start:end]

        console.print(Panel(
            Markdown(section),
            title=f"CSM Briefing -- {csm_name}",
            border_style="blue"
        ))
    else:
        console.print(f"[yellow]No briefing found for '{csm_name}'.[/yellow]")
        console.print("\nAvailable CSMs:")
        for line in content.split('\n'):
            if line.startswith('## '):
                console.print(f"  - {line[3:]}")


def list_accounts():
    """List all risk-scored accounts."""
    risk_results = load_risk_results()
    if risk_results is None:
        console.print("[red]No results found. Run 'python cli.py run' first.[/red]")
        return

    risk_results = risk_results.sort_values('final_risk_score', ascending=False)

    table = Table(title=f"All Renewal Accounts ({len(risk_results)} total)")
    table.add_column("ID", style="cyan", width=6)
    table.add_column("Account", width=28)
    table.add_column("Tier", width=10)
    table.add_column("Score", justify="center", width=7)
    table.add_column("ARR", justify="right", width=12)
    table.add_column("Days", justify="center", width=6)
    table.add_column("CSM", width=15)
    table.add_column("Plan", width=10)
    table.add_column("Confidence", width=10)

    for _, row in risk_results.iterrows():
        tier = row['risk_tier']
        color = {'High': 'red', 'Medium': 'yellow', 'Low': 'green'}.get(tier, 'white')

        table.add_row(
            str(int(row['account_id'])),
            str(row['account_name'])[:28],
            f"[{color}]{tier}[/{color}]",
            f"[{color}]{row['final_risk_score']:.0f}[/{color}]",
            f"${row['arr']:,.0f}",
            str(int(row['days_until_renewal'])),
            str(row['csm_name']),
            str(row['plan_tier']),
            str(row['confidence'])
        )

    console.print(table)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Renewal Intelligence Engine -- CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python cli.py run                      Run full pipeline
    python cli.py run --provider groq      Run with Groq (default)
    python cli.py run --provider openrouter  Run with OpenRouter
    python cli.py summary                  Show executive summary
    python cli.py list                     List all scored accounts
    python cli.py account 1007             Show Zenith Publishing detail
    python cli.py csm "Sarah Chen"         Show CSM briefing
        """
    )

    subparsers = parser.add_subparsers(dest='command')

    run_parser = subparsers.add_parser('run', help='Run full pipeline')
    run_parser.add_argument(
        '--provider', default='groq',
        choices=['groq', 'openrouter'],
        help='LLM provider'
    )

    subparsers.add_parser('summary', help='Show executive summary')
    subparsers.add_parser('list', help='List all scored accounts')

    account_parser = subparsers.add_parser('account', help='Show account detail')
    account_parser.add_argument('account_id', type=int)

    csm_parser = subparsers.add_parser('csm', help='Show CSM briefing')
    csm_parser.add_argument('csm_name', type=str)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    if args.command == 'run':
        run_full_pipeline(provider=args.provider)
    elif args.command == 'summary':
        display_summary()
    elif args.command == 'list':
        list_accounts()
    elif args.command == 'account':
        show_account_detail(args.account_id)
    elif args.command == 'csm':
        show_csm_briefing(args.csm_name)


if __name__ == "__main__":
    main()