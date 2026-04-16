"""
Renewal Intelligence Engine -- Streamlit Dashboard
Run: streamlit run app.py
"""

import sys
sys.path.insert(0, '.')

import json
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

from src.config import OUTPUT_DIR, REFERENCE_DATE, RENEWAL_CUTOFF_DATE

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Renewal Intelligence Engine",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# HELPERS
# ============================================================

def safe_parse_json(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def read_file_utf8(path: Path) -> str:
    """Read file with UTF-8 encoding."""
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_risk_results():
    path = OUTPUT_DIR / 'risk_scored_accounts.csv'
    if not path.exists():
        return None
    return pd.read_csv(path, encoding='utf-8-sig')


@st.cache_data
def load_executive_summary():
    path = OUTPUT_DIR / 'executive_summary.md'
    if not path.exists():
        return None
    return read_file_utf8(path)


@st.cache_data
def load_narratives():
    path = OUTPUT_DIR / 'account_narratives.md'
    if not path.exists():
        return None
    return read_file_utf8(path)


@st.cache_data
def load_csm_briefings():
    path = OUTPUT_DIR / 'csm_briefings.md'
    if not path.exists():
        return None
    return read_file_utf8(path)


@st.cache_data
def load_portfolio_insights():
    path = OUTPUT_DIR / 'portfolio_insights.json'
    if not path.exists():
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================================
# RUN PIPELINE IN UI
# ============================================================

def run_pipeline_ui():
    st.subheader("Run Pipeline")
    st.info(
        "This will run the complete pipeline: "
        "data ingestion → reconciliation → features → LLM analysis → risk scoring → explanations.\n\n"
        "Estimated time: 5-10 minutes."
    )

    provider = st.selectbox("LLM Provider", ["groq", "openrouter"])

    if st.button("Run Full Pipeline", type="primary"):
        try:
            from src.data_ingestion import load_all_data
            from src.data_reconciliation import reconcile_all_data
            from src.feature_engineering import build_feature_matrix
            from src.llm_pipeline import run_llm_pipeline
            from src.risk_scoring import score_all_accounts
            from src.explanation_generator import generate_all_explanations

            progress = st.progress(0, "Loading data...")
            data = load_all_data()

            progress.progress(15, "Reconciling data...")
            reconciled = reconcile_all_data(
                accounts_df=data['accounts'],
                usage_df=data['usage_metrics'],
                support_df=data['support_tickets'],
                csm_notes_raw=data['csm_notes_raw'],
                nps_df=data['nps_responses'],
                changelog_raw=data['changelog_raw'],
                llm_provider=provider
            )

            progress.progress(30, "Computing features...")
            all_features, renewal_features = build_feature_matrix(reconciled)

            progress.progress(45, "Running LLM analysis...")
            llm_results = run_llm_pipeline(
                reconciled_data=reconciled,
                feature_matrix=all_features,
                provider=provider
            )

            progress.progress(65, "Scoring accounts with LangGraph...")
            risk_results = score_all_accounts(
                feature_matrix=all_features,
                reconciled_data=reconciled,
                llm_results=llm_results
            )

            progress.progress(85, "Generating explanations...")
            generate_all_explanations(
                risk_results=risk_results,
                portfolio_insights=llm_results.get('portfolio_insights', {}),
                provider=provider
            )

            progress.progress(100, "Complete!")
            st.success("Pipeline complete! Refresh the page to see results.")
            st.cache_data.clear()

        except Exception as e:
            st.error(f"Pipeline failed: {str(e)}")
            st.exception(e)


# ============================================================
# PAGES
# ============================================================

def page_overview(risk_results: pd.DataFrame):
    st.title("Renewal Intelligence Engine")
    st.caption(
        f"Reference Date: {REFERENCE_DATE} | "
        f"Window: {REFERENCE_DATE} to {RENEWAL_CUTOFF_DATE}"
    )

    total_arr = risk_results['arr'].sum()
    high_risk = risk_results[risk_results['risk_tier'] == 'High']
    medium_risk = risk_results[risk_results['risk_tier'] == 'Medium']
    low_risk = risk_results[risk_results['risk_tier'] == 'Low']

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Accounts", len(risk_results))
    col2.metric("Total ARR", f"${total_arr:,.0f}")
    col3.metric(
        "High Risk",
        f"{len(high_risk)}",
        delta=f"-${high_risk['arr'].sum():,.0f} ARR at risk",
        delta_color="inverse"
    )
    col4.metric("Medium Risk", f"{len(medium_risk)}")
    col5.metric("Low Risk", f"{len(low_risk)}")

    st.divider()

    color_map = {'High': '#FF4B4B', 'Medium': '#FFA500', 'Low': '#00CC66'}

    col_left, col_right = st.columns(2)

    with col_left:
        tier_data = risk_results.groupby('risk_tier').agg(
            count=('account_id', 'count'),
            arr=('arr', 'sum')
        ).reset_index()

        fig_pie = px.pie(
            tier_data, values='arr', names='risk_tier',
            title='ARR by Risk Tier', color='risk_tier',
            color_discrete_map=color_map, hole=0.4
        )
        fig_pie.update_traces(
            textinfo='percent+value',
            texttemplate='%{percent}<br>$%{value:,.0f}'
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        fig_hist = px.histogram(
            risk_results, x='final_risk_score', nbins=20,
            title='Risk Score Distribution', color='risk_tier',
            color_discrete_map=color_map,
            labels={'final_risk_score': 'Risk Score'}
        )
        fig_hist.add_vline(
            x=70, line_dash="dash", line_color="red",
            annotation_text="High threshold (70)"
        )
        fig_hist.add_vline(
            x=40, line_dash="dash", line_color="orange",
            annotation_text="Medium threshold (40)"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    fig_scatter = px.scatter(
        risk_results, x='final_risk_score', y='arr',
        color='risk_tier', color_discrete_map=color_map,
        size='arr', hover_data=['account_name', 'csm_name', 'days_until_renewal'],
        title='ARR vs Risk Score (bubble size = ARR)',
        labels={'final_risk_score': 'Risk Score', 'arr': 'ARR ($)'}
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        csm_data = risk_results.groupby('csm_name').agg(
            accounts=('account_id', 'count'),
            arr=('arr', 'sum'),
            avg_risk=('final_risk_score', 'mean'),
            high_risk=('risk_tier', lambda x: (x == 'High').sum())
        ).sort_values('avg_risk', ascending=False).reset_index()

        fig_csm = px.bar(
            csm_data, x='csm_name', y='avg_risk',
            color='high_risk',
            title='Average Risk Score by CSM',
            labels={
                'csm_name': 'CSM',
                'avg_risk': 'Avg Risk Score',
                'high_risk': 'High Risk Accounts'
            },
            color_continuous_scale='Reds'
        )
        st.plotly_chart(fig_csm, use_container_width=True)

    with col2:
        timeline_data = risk_results.copy()
        timeline_data['contract_end_date'] = pd.to_datetime(
            timeline_data['contract_end_date']
        )

        fig_timeline = px.scatter(
            timeline_data, x='contract_end_date', y='final_risk_score',
            color='risk_tier', color_discrete_map=color_map,
            size='arr', hover_data=['account_name', 'arr'],
            title='Renewal Timeline vs Risk Score',
            labels={
                'contract_end_date': 'Contract End Date',
                'final_risk_score': 'Risk Score'
            }
        )
        st.plotly_chart(fig_timeline, use_container_width=True)


def page_high_risk(risk_results: pd.DataFrame):
    st.title("High Risk Accounts")

    high_risk = risk_results[risk_results['risk_tier'] == 'High'].sort_values(
        'final_risk_score', ascending=False
    )

    if len(high_risk) == 0:
        st.info("No high risk accounts identified.")
        return

    st.metric(
        "High Risk ARR",
        f"${high_risk['arr'].sum():,.0f}",
        delta=f"{len(high_risk)} accounts",
        delta_color="inverse"
    )

    for _, row in high_risk.iterrows():
        with st.expander(
            f"{row['account_name']} -- Score: {row['final_risk_score']:.0f} | "
            f"ARR: ${row['arr']:,.0f} | {int(row['days_until_renewal'])} days",
            expanded=True
        ):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Risk Score", f"{row['final_risk_score']:.0f}/100")
            col2.metric("ARR", f"${row['arr']:,.0f}")
            col3.metric("Days Left", int(row['days_until_renewal']))
            col4.metric("Confidence", row['confidence'])

            st.subheader("Health Scores")
            health_cols = st.columns(5)

            for i, (label, key) in enumerate([
                ('Usage', 'usage_health'), ('Support', 'support_health'),
                ('NPS', 'nps_health'), ('CSM', 'csm_health'), ('SDK', 'sdk_health')
            ]):
                val = float(row[key])
                health_cols[i].metric(label, f"{val:.0f}/100")
                health_cols[i].progress(val / 100)

            col_left, col_right = st.columns(2)

            with col_left:
                st.subheader("Risk Factors")
                factors = safe_parse_json(row.get('risk_factors'))
                for f in factors:
                    sev = f.get('severity', '?')
                    sev_label = {
                        'critical': '**[CRITICAL]**',
                        'high': '**[HIGH]**',
                        'medium': '[MEDIUM]',
                        'low': '[LOW]'
                    }.get(sev, f'[{sev}]')
                    st.write(
                        f"{sev_label} **{f.get('factor', '')}**: {f.get('detail', '')}"
                    )

            with col_right:
                st.subheader("Recommended Actions")
                actions = safe_parse_json(row.get('recommended_actions'))
                for a in actions:
                    st.write(f"→ {a}")

            if row.get('competitors'):
                st.warning(f"Competitors mentioned: **{row['competitors']}**")

            st.caption(
                f"CSM: {row['csm_name']} | Plan: {row['plan_tier']} | "
                f"Industry: {row['industry']} | Region: {row['region']}"
            )


def page_account_lookup(risk_results: pd.DataFrame):
    st.title("Account Lookup")

    sorted_results = risk_results.sort_values('final_risk_score', ascending=False)
    account_options = {
        f"{int(row['account_id'])} -- {row['account_name']} ({row['risk_tier']})": int(row['account_id'])
        for _, row in sorted_results.iterrows()
    }

    selected = st.selectbox("Select Account", list(account_options.keys()))

    if not selected:
        return

    account_id = account_options[selected]
    row = risk_results[risk_results['account_id'] == account_id].iloc[0]

    tier = row['risk_tier']
    tier_color = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}.get(tier, 'gray')

    st.header(f"{row['account_name']}")
    st.caption(
        f"ID: {account_id} | Plan: {row['plan_tier']} | "
        f"Industry: {row['industry']} | CSM: {row['csm_name']}"
    )

    cols = st.columns(6)
    cols[0].metric("Risk Score", f"{row['final_risk_score']:.0f}/100")
    cols[1].metric("Risk Tier", tier)
    cols[2].metric("ARR", f"${row['arr']:,.0f}")
    cols[3].metric("Days Left", int(row['days_until_renewal']))
    cols[4].metric("Confidence", row['confidence'])
    cols[5].metric("Relationship", row.get('relationship_health', 'unknown'))

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        # Waterfall chart
        fig_waterfall = go.Figure(go.Waterfall(
            orientation="v",
            x=["Quantitative", "Qualitative Adj.", "Conflict Adj.", "Final Score"],
            y=[
                float(row['quantitative_score']),
                float(row['qualitative_adjustment']),
                float(row['conflict_adjustment']),
                0
            ],
            measure=["absolute", "relative", "relative", "total"],
            text=[
                f"{row['quantitative_score']:.0f}",
                f"{row['qualitative_adjustment']:+.0f}",
                f"{row['conflict_adjustment']:+.0f}",
                f"{row['final_risk_score']:.0f}"
            ],
            connector={"line": {"color": "rgb(63, 63, 63)"}}
        ))
        fig_waterfall.update_layout(title="Risk Score Breakdown", height=350)
        st.plotly_chart(fig_waterfall, use_container_width=True)

    with col_right:
        # Radar chart
        categories = ['Usage', 'Support', 'NPS', 'CSM', 'SDK']
        values = [
            float(row['usage_health']), float(row['support_health']),
            float(row['nps_health']), float(row['csm_health']),
            float(row['sdk_health'])
        ]

        line_color = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}.get(tier, 'blue')
        fill_color = {
            'High': 'rgba(255,0,0,0.1)',
            'Medium': 'rgba(255,165,0,0.1)',
            'Low': 'rgba(0,204,102,0.1)'
        }.get(tier, 'rgba(0,0,255,0.1)')

        fig_radar = go.Figure(data=go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            fillcolor=fill_color,
            line_color=line_color
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            title="Health Radar",
            height=350
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Detail tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Risk Factors", "Actions", "Narrative", "Raw Data"])

    with tab1:
        factors = safe_parse_json(row.get('risk_factors'))
        if factors:
            for f in factors:
                sev = f.get('severity', '?')
                sev_label = {
                    'critical': '**[CRITICAL]**', 'high': '**[HIGH]**',
                    'medium': '[MEDIUM]', 'low': '[LOW]'
                }.get(sev, f'[{sev}]')
                st.write(
                    f"{sev_label} **{f.get('factor', '')}** "
                    f"(impact: {f.get('score_impact', 0)})"
                )
                st.caption(f.get('detail', ''))
        else:
            st.info("No significant risk factors identified.")

    with tab2:
        actions = safe_parse_json(row.get('recommended_actions'))
        for i, a in enumerate(actions, 1):
            st.write(f"{i}. {a}")

    with tab3:
        narratives = load_narratives()
        if narratives:
            account_name = str(row['account_name'])
            sections = narratives.split("---")
            found = False
            for section in sections:
                if account_name in section:
                    st.markdown(section.strip())
                    found = True
                    break
            if not found:
                st.info("No narrative generated for this account.")
        else:
            st.info("Run the pipeline first to generate narratives.")

    with tab4:
        st.json(row.to_dict())


def page_executive_summary():
    st.title("Executive Summary")

    summary = load_executive_summary()
    if summary:
        st.markdown(summary)
    else:
        st.info("No executive summary available. Run the pipeline first.")

    insights = load_portfolio_insights()
    if insights and 'portfolio_insights' in insights:
        st.divider()
        st.subheader("Non-Obvious Insights")

        for insight in insights['portfolio_insights']:
            urgency = insight.get('urgency', 'unknown')
            urgency_map = {
                'immediate': '[IMMEDIATE]', 'this_week': '[THIS WEEK]',
                'this_month': '[THIS MONTH]', 'this_quarter': '[THIS QUARTER]'
            }
            urgency_label = urgency_map.get(urgency, f'[{urgency.upper()}]')

            with st.expander(
                f"{urgency_label} {insight.get('insight_title', 'Insight')}",
                expanded=True
            ):
                st.write(insight.get('insight_detail', ''))
                st.info(f"**Evidence:** {insight.get('evidence', 'N/A')}")
                st.warning(f"**ARR at Risk:** {insight.get('arr_at_risk', 'Unknown')}")
                st.success(f"**Action:** {insight.get('recommended_action', 'N/A')}")
                st.caption(f"Urgency: {urgency_label}")

        if 'top_priority_action' in insights:
            st.error(f"**Top Priority:** {insights['top_priority_action']}")


def page_csm_view(risk_results: pd.DataFrame):
    st.title("CSM View")

    csm_names = sorted(risk_results['csm_name'].unique())
    selected_csm = st.selectbox("Select CSM", csm_names)

    if not selected_csm:
        return

    csm_accounts = risk_results[risk_results['csm_name'] == selected_csm].sort_values(
        'final_risk_score', ascending=False
    )

    cols = st.columns(4)
    cols[0].metric("Accounts", len(csm_accounts))
    cols[1].metric("Total ARR", f"${csm_accounts['arr'].sum():,.0f}")
    cols[2].metric("Avg Risk", f"{csm_accounts['final_risk_score'].mean():.0f}")
    cols[3].metric(
        "High Risk",
        len(csm_accounts[csm_accounts['risk_tier'] == 'High'])
    )

    briefings = load_csm_briefings()
    if briefings:
        marker = f"## {selected_csm}"
        if marker in briefings:
            start = briefings.index(marker)
            end = briefings.find("\n---\n", start)
            if end == -1:
                end = len(briefings)
            st.markdown(briefings[start:end])

    st.divider()
    st.subheader(f"{selected_csm}'s Accounts")

    display_df = csm_accounts[[
        'account_id', 'account_name', 'risk_tier', 'final_risk_score',
        'arr', 'days_until_renewal', 'plan_tier', 'confidence'
    ]].copy()
    display_df.columns = [
        'ID', 'Account', 'Tier', 'Score', 'ARR', 'Days', 'Plan', 'Confidence'
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            'ARR': st.column_config.NumberColumn(format="$%d"),
            'Score': st.column_config.ProgressColumn(
                min_value=0, max_value=100, format="%d"
            )
        }
    )


# ============================================================
# MAIN APP
# ============================================================

def main():
    risk_results = load_risk_results()

    st.sidebar.title("Renewal Intelligence")
    st.sidebar.caption(f"v1.0 | Ref: {REFERENCE_DATE}")

    if risk_results is None:
        st.sidebar.warning("No results found")
        run_pipeline_ui()
        return

    page = st.sidebar.radio(
        "Navigation",
        [
            "Overview",
            "High Risk",
            "Account Lookup",
            "Executive Summary",
            "CSM View",
            "Run Pipeline"
        ]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Filters")

    tier_filter = st.sidebar.multiselect(
        "Risk Tier", ['High', 'Medium', 'Low'],
        default=['High', 'Medium', 'Low']
    )
    csm_filter = st.sidebar.multiselect(
        "CSM", sorted(risk_results['csm_name'].unique()),
        default=sorted(risk_results['csm_name'].unique())
    )
    plan_filter = st.sidebar.multiselect(
        "Plan Tier", sorted(risk_results['plan_tier'].unique()),
        default=sorted(risk_results['plan_tier'].unique())
    )

    filtered = risk_results[
        (risk_results['risk_tier'].isin(tier_filter)) &
        (risk_results['csm_name'].isin(csm_filter)) &
        (risk_results['plan_tier'].isin(plan_filter))
    ]

    if page == "Overview":
        page_overview(filtered)
    elif page == "High Risk":
        page_high_risk(filtered)
    elif page == "Account Lookup":
        page_account_lookup(risk_results)
    elif page == "Executive Summary":
        page_executive_summary()
    elif page == "CSM View":
        page_csm_view(filtered)
    elif page == "Run Pipeline":
        run_pipeline_ui()


if __name__ == "__main__":
    main()