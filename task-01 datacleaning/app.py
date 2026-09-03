import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pipeline import DataCleaningPipeline

# Page Configuration
st.set_page_config(
    page_title="Task-01 | Data Cleaning & Preprocessing Studio",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Modern CSS Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Hero Banner */
    .hero-container {
        background: linear-gradient(135deg, #1E1B4B 0%, #312E81 50%, #4338CA 100%);
        padding: 30px 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(49, 46, 129, 0.25);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #C7D2FE;
        margin-top: 8px;
        margin-bottom: 16px;
        max-width: 850px;
        line-height: 1.5;
    }
    .hero-badges {
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }
    .badge {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(8px);
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    /* Metric Card Styling */
    .stat-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .stat-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stat-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin: 4px 0;
    }
    .stat-change {
        font-size: 0.85rem;
        font-weight: 600;
    }
    .text-positive { color: #10B981; }
    .text-negative { color: #EF4444; }
    .text-neutral { color: #64748B; }

    /* Modern Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        font-weight: 600;
        border-radius: 8px;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EEF2FF !important;
        color: #4F46E5 !important;
    }

    /* Sidebar Styling */
    .sidebar-header {
        display: flex;
        align-items: center;
        gap: 12px;
        padding-bottom: 15px;
        margin-bottom: 15px;
        border-bottom: 1px solid #E2E8F0;
    }
    .sidebar-logo {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #4F46E5, #06B6D4);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        color: white;
        box-shadow: 0 4px 10px rgba(79, 70, 229, 0.3);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_dataset():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv_path = os.path.join(current_dir, "student_performance_updated_1000.csv")
    if os.path.exists(default_csv_path):
        return pd.read_csv(default_csv_path)
    return None


def read_uploaded_file(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xls', '.xlsx')):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format! Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None


# ===================== SIDEBAR CONTROLS =====================
with st.sidebar:
    # Reliable modern SVG / HTML Logo that never breaks
    st.markdown("""
    <div class="sidebar-header">
        <div class="sidebar-logo">🧹</div>
        <div>
            <h3 style="margin: 0; font-size: 1.15rem; font-weight: 700; color: #1E293B;">CleanStudio</h3>
            <span style="font-size: 0.78rem; font-weight: 600; color: #6366F1; background: #EEF2FF; padding: 2px 8px; border-radius: 6px;">TASK 01 • INCODEVISION</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Select Dataset:",
        ["🎯 Default Dataset (Student Performance)", "📤 Upload Custom File (CSV/Excel)"],
        label_visibility="collapsed"
    )

    raw_df = None
    dataset_name = ""

    if data_source.startswith("🎯"):
        raw_df = load_default_dataset()
        dataset_name = "student_performance_updated_1000.csv"
        if raw_df is not None:
            st.success("✅ Default dataset loaded (1000 rows)")
        else:
            st.error("Default dataset not found.")
    else:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel dataset:",
            type=["csv", "xlsx", "xls"],
            help="Upload any tabular dataset containing missing values or duplicates."
        )
        if uploaded_file is not None:
            raw_df = read_uploaded_file(uploaded_file)
            dataset_name = uploaded_file.name
            if raw_df is not None:
                st.success(f"✅ Loaded: {uploaded_file.name} ({len(raw_df)} rows)")
        else:
            st.info("Upload a file above to begin cleaning.")

    st.markdown("---")
    st.markdown("### ⚙️ Pipeline Settings")

    pipeline_mode = st.radio(
        "Execution Mode:",
        ["⚡ 1-Click Auto Clean (Recommended)", "🛠️ Custom Pipeline Controls"]
    )

    if pipeline_mode.startswith("⚡"):
        st.caption("✨ Smart defaults: Removes duplicates, fixes formats, imputes missing values with median/mode, and winsorizes outliers.")
        remove_dups = True
        dup_subset = None
        fix_formats = True
        impute_nulls = True
        numeric_strategy = "median"
        categorical_strategy = "mode"
        handle_outliers = True
        outlier_method = "iqr"
        outlier_action = "clip"
        encode_cats = False
        encoding_method = "label"
        scale_nums = False
        scaling_method = "standard"
    else:
        with st.expander("1. 🔁 Duplicate Records", expanded=True):
            remove_dups = st.checkbox("Remove Duplicates", value=True)
            if raw_df is not None and remove_dups:
                subset_on = st.checkbox("By specific key columns only", value=False)
                if subset_on:
                    dup_subset = st.multiselect("Select key columns:", raw_df.columns.tolist())
                else:
                    dup_subset = None
            else:
                dup_subset = None

        with st.expander("2. 🔤 Formats & Cleaning", expanded=False):
            fix_formats = st.checkbox("Trim whitespaces & clean strings", value=True)
            auto_booleans = st.checkbox("Standardize booleans & numbers", value=True)

        with st.expander("3. 🩹 Missing Value Imputation", expanded=True):
            impute_nulls = st.checkbox("Handle Missing Values", value=True)
            numeric_strategy = st.selectbox(
                "Numeric Columns Strategy:",
                ["median", "mean", "mode", "zero", "drop"],
                index=0
            )
            categorical_strategy = st.selectbox(
                "Categorical Columns Strategy:",
                ["mode", "unknown", "drop"],
                index=0
            )

        with st.expander("4. 🎯 Outlier Treatment", expanded=False):
            handle_outliers = st.checkbox("Detect & Handle Outliers", value=True)
            outlier_method = st.selectbox("Detection Method:", ["iqr", "zscore"])
            outlier_action = st.selectbox("Action on Outliers:", ["clip", "drop", "none"])

        with st.expander("5. 🤖 ML Feature Preprocessing", expanded=False):
            encode_cats = st.checkbox("Encode Categorical Variables", value=False)
            encoding_method = st.selectbox("Encoding Method:", ["label", "onehot"])
            scale_nums = st.checkbox("Scale Numerical Variables", value=False)
            scaling_method = st.selectbox("Scaling Method:", ["standard", "minmax", "robust"])


# ===================== MAIN CONTENT AREA =====================

# Modern Hero Banner
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Universal Data Cleaning & Preprocessing Studio</div>
    <div class="hero-subtitle">
        An intelligent, automated pipeline designed to audit, clean, standardize, and prepare raw tabular datasets for exploratory analysis and Machine Learning modeling.
    </div>
    <div class="hero-badges">
        <span class="badge">🚀 Automated Pipeline</span>
        <span class="badge">🧹 Missing Value Imputation</span>
        <span class="badge">🔁 Duplicate Elimination</span>
        <span class="badge">🎯 IQR Outlier Capping</span>
        <span class="badge">📊 Plotly Interactive Visuals</span>
    </div>
</div>
""", unsafe_allow_html=True)

if raw_df is None:
    st.info("👈 Please select the default dataset or upload your custom dataset from the sidebar to start.")
    st.stop()

# Initialize & Run Pipeline
pipeline = DataCleaningPipeline()
initial_health = pipeline.get_dataset_health(raw_df)

cleaned_df, audit_log, summary = pipeline.run_pipeline(
    raw_df,
    remove_dups=remove_dups,
    dup_subset=dup_subset if dup_subset else None,
    fix_formats=fix_formats,
    impute_nulls=impute_nulls,
    numeric_strategy=numeric_strategy,
    categorical_strategy=categorical_strategy,
    handle_outliers=handle_outliers,
    outlier_method=outlier_method,
    outlier_action=outlier_action,
    encode_cats=encode_cats,
    encoding_method=encoding_method,
    scale_nums=scale_nums,
    scaling_method=scaling_method
)
final_health = summary["final"]

# Top Interactive Metric Cards
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Rows</div>
        <div class="stat-value">{final_health['total_rows']:,}</div>
        <div class="stat-change text-neutral">Initial: {initial_health['total_rows']:,}</div>
    </div>
    """, unsafe_allow_html=True)

with m_col2:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Total Columns</div>
        <div class="stat-value">{final_health['total_cols']:,}</div>
        <div class="stat-change text-neutral">Features ready</div>
    </div>
    """, unsafe_allow_html=True)

with m_col3:
    fixed_nulls = initial_health['total_missing_values'] - final_health['total_missing_values']
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Missing Values</div>
        <div class="stat-value">{final_health['total_missing_values']}</div>
        <div class="stat-change text-positive">✓ {fixed_nulls:,} fixed</div>
    </div>
    """, unsafe_allow_html=True)

with m_col4:
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Duplicates Left</div>
        <div class="stat-value">{final_health['duplicate_rows']}</div>
        <div class="stat-change text-positive">✓ Cleaned</div>
    </div>
    """, unsafe_allow_html=True)

with m_col5:
    health_score = 100 if final_health['total_missing_values'] == 0 else round(100 - final_health['missing_percentage'], 1)
    st.markdown(f"""
    <div class="stat-card">
        <div class="stat-label">Data Quality Score</div>
        <div class="stat-value">{health_score}%</div>
        <div class="stat-change text-positive">✓ Optimal</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# Main Navigation Tabs
tab_explore, tab_compare, tab_audit, tab_export = st.tabs([
    "📊 Raw Data Diagnostics",
    "✨ Cleaned Data & Visuals",
    "📋 Pipeline Audit Trail",
    "📥 Export & Download"
])

# ----------------- TAB 1: RAW DATA DIAGNOSTICS -----------------
with tab_explore:
    st.subheader(f"Raw Dataset Inspection: `{dataset_name}`")

    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("**Sample Raw Records (First 8 Rows)**")
        st.dataframe(raw_df.head(8), use_container_width=True)

    with c2:
        st.markdown("**Data Completeness & Null Analysis**")
        missing_df = pd.DataFrame(initial_health["columns"])
        missing_filtered = missing_df[missing_df["Null Count"] > 0].sort_values(by="Null Count", ascending=True)
        
        total_cells = initial_health['total_rows'] * initial_health['total_cols']
        missing_cells = initial_health['total_missing_values']
        valid_cells = total_cells - missing_cells
        health_pct = round((valid_cells / total_cells) * 100, 1) if total_cells > 0 else 100.0

        chart_choice = st.radio(
            "Select View Mode:",
            ["🍩 Data Health Overview (Donut Chart)", "📊 Detailed Feature Breakdown (Horizontal)"],
            horizontal=True,
            label_visibility="collapsed"
        )

        if chart_choice.startswith("🍩"):
            fig_donut = go.Figure(data=[go.Pie(
                labels=['Valid Cells', 'Missing Cells'],
                values=[valid_cells, missing_cells],
                hole=0.68,
                marker=dict(colors=['#10B981', '#EF4444']),
                textinfo='percent',
                textfont=dict(size=13, color='#FFFFFF'),
                hoverinfo='label+value+percent'
            )])
            fig_donut.update_layout(
                annotations=[dict(
                    text=f"<b>{health_pct}%</b><br><span style='font-size:11px; color:#64748B;'>Complete</span>",
                    x=0.5, y=0.5, font_size=18, showarrow=False
                )],
                margin=dict(l=10, r=10, t=10, b=10),
                height=290,
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            if not missing_filtered.empty:
                fig_horiz = go.Figure(go.Bar(
                    x=missing_filtered["Null Count"],
                    y=missing_filtered["Column"],
                    orientation='h',
                    marker=dict(
                        color='#6366F1',
                        line=dict(color='#4F46E5', width=1)
                    ),
                    text=missing_filtered.apply(lambda r: f"{r['Null Count']} ({r['Null %']}%)", axis=1),
                    textposition='outside',
                ))
                fig_horiz.update_layout(
                    margin=dict(l=10, r=40, t=15, b=10),
                    height=290,
                    xaxis_title="Null Count",
                    yaxis=dict(autorange="reversed"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig_horiz, use_container_width=True)
            else:
                st.success("🎉 No missing values found in the dataset!")

    st.markdown("**Feature Schema & Data Quality Details**")
    st.dataframe(pd.DataFrame(initial_health["columns"]), use_container_width=True)


# ----------------- TAB 2: CLEANED DATA & VISUALS -----------------
with tab_compare:
    st.subheader("Cleaned & Standardized Dataset")
    st.dataframe(cleaned_df.head(10), use_container_width=True)

    st.markdown("---")
    st.subheader("Before vs. After Visual Comparison")

    v_col1, v_col2 = st.columns(2)

    with v_col1:
        comp_df = pd.DataFrame({
            "Stage": ["Raw Dataset", "Cleaned Dataset"],
            "Missing Values": [initial_health["total_missing_values"], final_health["total_missing_values"]],
            "Duplicates": [initial_health["duplicate_rows"], final_health["duplicate_rows"]]
        })
        fig_comp = px.bar(
            comp_df,
            x="Stage",
            y=["Missing Values", "Duplicates"],
            barmode="group",
            title="Data Integrity: Before vs. After",
            color_discrete_sequence=["#EF4444", "#10B981"]
        )
        fig_comp.update_layout(
            height=340,
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig_comp, use_container_width=True)

    with v_col2:
        numeric_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            selected_feature = st.selectbox("Select Feature to Compare Distribution:", numeric_cols)
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=raw_df[selected_feature].dropna(),
                name="Before Cleaning",
                opacity=0.6,
                marker_color="#F43F5E"
            ))
            fig_dist.add_trace(go.Histogram(
                x=cleaned_df[selected_feature].dropna() if selected_feature in cleaned_df.columns else [],
                name="After Cleaning",
                opacity=0.6,
                marker_color="#10B981"
            ))
            fig_dist.update_layout(
                barmode='overlay',
                title=f"Distribution: {selected_feature}",
                height=340,
                margin=dict(l=20, r=20, t=40, b=20),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# ----------------- TAB 3: AUDIT TRAIL -----------------
with tab_audit:
    st.subheader("Step-by-Step Pipeline Audit Log")
    st.caption("A chronological record of every data cleaning and transformation step executed.")

    for i, step in enumerate(audit_log, start=1):
        with st.expander(f"Step {i}: {step['step']}", expanded=(i <= 2)):
            st.markdown(f"**Action Performed:** {step['details']}")
            if step['stats']:
                st.json(step['stats'])


# ----------------- TAB 4: EXPORT & DOWNLOAD -----------------
with tab_export:
    st.subheader("Export Cleaned Data & Pipeline Report")
    st.markdown("Download your cleaned dataset ready for training ML models or visualizing in PowerBI / Tableau.")

    d_col1, d_col2 = st.columns(2)

    with d_col1:
        csv_buffer = io.StringIO()
        cleaned_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')

        clean_filename = f"cleaned_{dataset_name if dataset_name.endswith('.csv') else dataset_name + '.csv'}"
        st.download_button(
            label="⬇️ Download Cleaned CSV Dataset",
            data=csv_data,
            file_name=clean_filename,
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

    with d_col2:
        report_data = {
            "dataset": dataset_name,
            "initial_health": initial_health,
            "final_health": final_health,
            "audit_trail": audit_log
        }
        json_report = json.dumps(report_data, indent=2, default=str)

        st.download_button(
            label="📄 Download Full Audit Report (JSON)",
            data=json_report,
            file_name=f"audit_report_{dataset_name}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### 💻 Reproduce in Python Code")
    st.code(f"""from pipeline import DataCleaningPipeline
import pandas as pd

# 1. Load dataset
df = pd.read_csv('{dataset_name}')

# 2. Initialize and run cleaning pipeline
pipeline = DataCleaningPipeline()
clean_df, audit_log, summary = pipeline.run_pipeline(
    df,
    remove_dups=True,
    impute_nulls=True,
    handle_outliers=True
)

# 3. Export cleaned dataset
clean_df.to_csv('{clean_filename}', index=False)
print("Pipeline executed successfully!")
""", language="python")
