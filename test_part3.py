"""
Test Part 3 — Feature Engineering.
Run: python test_part3.py
"""

import sys
sys.path.insert(0, '.')

import pandas as pd
from rich.console import Console
from rich.table import Table

from src.data_ingestion import load_all_data
from src.data_reconciliation import reconcile_all_data
from src.feature_engineering import build_feature_matrix

console = Console()


def test_full_pipeline():
    """Run ingestion → reconciliation → feature engineering."""

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

    assert len(all_features) > 0, "Feature matrix is empty!"
    assert len(renewal_features) > 0, "No renewal accounts found!"

    console.print(
        f"\n[bold green]Feature matrix built: {len(renewal_features)} "
        f"renewal accounts, {len(all_features.columns)} features[/bold green]"
    )

    # Display health score table
    table = Table(title="Renewal Accounts -- Health Score Summary")
    table.add_column("ID", style="cyan")
    table.add_column("Account", style="white")
    table.add_column("ARR", style="green", justify="right")
    table.add_column("Days", justify="right")
    table.add_column("Usage", justify="center")
    table.add_column("Support", justify="center")
    table.add_column("NPS", justify="center")
    table.add_column("CSM", justify="center")
    table.add_column("SDK", justify="center")

    def color_score(score):
        if score is None or pd.isna(score):
            return "[dim]N/A[/dim]"
        score = float(score)
        if score >= 70:
            return f"[green]{score:.0f}[/green]"
        elif score >= 40:
            return f"[yellow]{score:.0f}[/yellow]"
        else:
            return f"[red]{score:.0f}[/red]"

    display_df = renewal_features.sort_values(
        'usage_health_score', ascending=True
    ).head(30)

    for _, row in display_df.iterrows():
        table.add_row(
            str(row['account_id']),
            str(row['account_name'])[:25],
            f"${row['arr']:,.0f}",
            str(int(row['days_until_renewal'])),
            color_score(row.get('usage_health_score')),
            color_score(row.get('support_health_score')),
            color_score(row.get('nps_health_score')),
            color_score(row.get('csm_health_score')),
            color_score(row.get('sdk_health_score'))
        )

    console.print(table)

    # Feature group counts
    console.print(f"\n[bold]Feature groups:[/bold]")
    groups = {
        'Contract': [c for c in all_features.columns if c in [
            'arr', 'plan_tier', 'days_until_renewal', 'renewal_urgency',
            'arr_tier', 'plan_tier_numeric', 'arr_plan_mismatch', 'in_renewal_window'
        ]],
        'Usage': [c for c in all_features.columns if any(
            c.startswith(p) for p in [
                'api_calls', 'content_entries', 'active_users',
                'workflows', 'usage_health', 'declining', 'zero_', 'workflow_abandoned'
            ]
        )],
        'Support': [c for c in all_features.columns if any(
            c.startswith(p) for p in [
                'total_ticket', 'p1_', 'p2_', 'p3_', 'p4_', 'open_',
                'escalat', 'avg_res', 'max_res', 'recent_ticket',
                'ticket_vel', 'has_dep', 'has_mig', 'has_block',
                'has_recur', 'support_health', 'no_ticket'
            ]
        )],
        'NPS': [c for c in all_features.columns if c.startswith('nps_')],
        'CSM': [c for c in all_features.columns if c.startswith('csm_')],
        'SDK': [c for c in all_features.columns if c.startswith('sdk_')],
        'Conflicts': [c for c in all_features.columns if any(
            c.startswith(p) for p in [
                'conflict', 'has_conflict', 'high_sev',
                'has_nps_contr', 'has_nps_csm', 'has_csm_supp'
            ]
        )]
    }

    for group_name, cols in groups.items():
        console.print(f"  {group_name}: {len(cols)} features")

    # Concerning accounts
    console.print(f"\n[bold red]Accounts with concerning signals:[/bold red]")
    concerning = renewal_features[
        (renewal_features['usage_health_score'] < 50) |
        (renewal_features['support_health_score'] < 50) |
        (renewal_features['csm_health_score'] < 30)
    ].sort_values('usage_health_score')

    for _, row in concerning.iterrows():
        flags = []
        if float(row.get('usage_health_score', 100)) < 50:
            flags.append(f"usage={row['usage_health_score']:.0f}")
        if float(row.get('support_health_score', 100)) < 50:
            flags.append(f"support={row['support_health_score']:.0f}")
        if float(row.get('csm_health_score', 100)) < 30:
            flags.append(f"csm={row['csm_health_score']:.0f}")
        if row.get('sdk_is_critical', False):
            flags.append("SDK_CRITICAL")
        if row.get('csm_competitor_count', 0) > 0:
            flags.append(f"competitors={row.get('csm_competitors_list', '')}")

        console.print(
            f"  [{row['account_id']}] {row['account_name']}: "
            f"{', '.join(flags)}"
        )

    console.print(
        f"\n[bold green]Saved to outputs/all_account_features.csv "
        f"and outputs/renewal_account_features.csv[/bold green]"
    )

    return all_features, renewal_features


if __name__ == "__main__":
    all_features, renewal_features = test_full_pipeline()