import os
import io
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pipeline import DataCleaningPipeline

# Set Page Config
st.set_page_config(
    page_title="Task-01: Data Cleaning & Preprocessing Pipeline",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean UI styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        font-weight: 600;
        border-radius: 6px;
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


# Sidebar Controls
st.sidebar.image("https://img.icons8.com/clouds/100/000000/data-backup.png", width=70)
st.sidebar.title("Pipeline Controls")

# Data Source Selection
data_source = st.sidebar.radio(
    "Select Dataset Source:",
    ["Default Task Dataset", "Upload Custom Dataset (CSV/Excel)"]
)

raw_df = None
dataset_name = ""

if data_source == "Default Task Dataset":
    raw_df = load_default_dataset()
    dataset_name = "student_performance_updated_1000.csv"
    if raw_df is None:
        st.sidebar.error("Default dataset not found in current folder.")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Upload your CSV or Excel file:",
        type=["csv", "xlsx", "xls"]
    )
    if uploaded_file is not None:
        raw_df = read_uploaded_file(uploaded_file)
        dataset_name = uploaded_file.name
    else:
        st.sidebar.info("Upload a dataset to run the pipeline.")

# Pipeline Configuration Mode
st.sidebar.markdown("---")
st.sidebar.subheader("Pipeline Configuration")

mode = st.sidebar.radio(
    "Choose Mode:",
    ["Auto-Clean (Recommended)", "Custom Pipeline Settings"]
)

if mode == "Auto-Clean (Recommended)":
    st.sidebar.caption("⚡ Auto-detects nulls, duplicates, formatting errors, and clips extreme outliers with smart statistical defaults.")
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
    with st.sidebar.expander("1. Duplicates Handling", expanded=True):
        remove_dups = st.checkbox("Remove Duplicate Rows", value=True)
        dup_subset_cols = []
        if raw_df is not None:
            use_subset = st.checkbox("Check duplicates by specific columns (e.g., ID)", value=False)
            if use_subset:
                dup_subset = st.multiselect("Select unique key columns:", raw_df.columns.tolist())
            else:
                dup_subset = None
        else:
            dup_subset = None

    with st.sidebar.expander("2. Formats & Data Types", expanded=False):
        fix_formats = st.checkbox("Trim whitespaces & clean strings", value=True)
        auto_booleans = st.checkbox("Auto-standardize booleans", value=True)

    with st.sidebar.expander("3. Missing Values Imputation", expanded=True):
        impute_nulls = st.checkbox("Handle Missing Values", value=True)
        numeric_strategy = st.selectbox(
            "Numeric Imputation Strategy:",
            ["median", "mean", "mode", "zero", "drop"]
        )
        categorical_strategy = st.selectbox(
            "Categorical Imputation Strategy:",
            ["mode", "unknown", "drop"]
        )

    with st.sidebar.expander("4. Outlier Handling", expanded=False):
        handle_outliers = st.checkbox("Detect and Treat Outliers", value=True)
        outlier_method = st.selectbox("Detection Method:", ["iqr", "zscore"])
        outlier_action = st.selectbox("Action on Outliers:", ["clip", "drop", "none"])

    with st.sidebar.expander("5. Machine Learning Preprocessing", expanded=False):
        encode_cats = st.checkbox("Encode Categorical Variables", value=False)
        encoding_method = st.selectbox("Encoding Method:", ["label", "onehot"])
        scale_nums = st.checkbox("Scale Numerical Variables", value=False)
        scaling_method = st.selectbox("Scaling Method:", ["standard", "minmax", "robust"])


# Main Content Area
st.markdown('<div class="main-header">Task-01: Data Cleaning & Preprocessing Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Automated, robust pipeline to clean, standardize, and prepare any dataset for analytics and Machine Learning.</div>', unsafe_allow_html=True)

if raw_df is None:
    st.info("👈 Please select or upload a dataset using the sidebar to begin.")
    st.stop()

# Initialize Pipeline Engine
pipeline = DataCleaningPipeline()
initial_health = pipeline.get_dataset_health(raw_df)

# Execute Pipeline
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

# Top Metrics Banner
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Rows", f"{final_health['total_rows']:,}", delta=f"{final_health['total_rows'] - initial_health['total_rows']:,} rows" if final_health['total_rows'] != initial_health['total_rows'] else "Unchanged")
col2.metric("Columns", f"{final_health['total_cols']:,}", delta=f"{final_health['total_cols'] - initial_health['total_cols']:,} cols" if final_health['total_cols'] != initial_health['total_cols'] else "Unchanged")
col3.metric("Missing Values", f"{final_health['total_missing_values']:,}", delta=f"-{initial_health['total_missing_values'] - final_health['total_missing_values']:,} fixed" if initial_health['total_missing_values'] > 0 else "0", delta_color="inverse")
col4.metric("Duplicates Left", f"{final_health['duplicate_rows']:,}", delta=f"-{initial_health['duplicate_rows']} removed" if initial_health['duplicate_rows'] > 0 else "0", delta_color="inverse")
col5.metric("Dataset Health", "100% Clean" if final_health['total_missing_values'] == 0 else f"{100 - final_health['missing_percentage']:.1f}%")

st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Raw Data & Diagnostics",
    "✨ Cleaned Dataset & Comparison",
    "📋 Pipeline Audit Report",
    "📥 Download & Export"
])

# TAB 1: Raw Data & Diagnostics
with tab1:
    st.subheader(f"Raw Dataset: `{dataset_name}`")
    
    col_t1, col_t2 = st.columns([3, 2])
    with col_t1:
        st.markdown("**Sample Raw Data (First 10 Rows)**")
        st.dataframe(raw_df.head(10), use_container_width=True)

    with col_t2:
        st.markdown("**Missing Values by Column (Raw)**")
        missing_df = pd.DataFrame(initial_health["columns"])
        missing_filtered = missing_df[missing_df["Null Count"] > 0]
        if not missing_filtered.empty:
            fig_missing = px.bar(
                missing_filtered,
                x="Column",
                y="Null Count",
                color="Null %",
                color_continuous_scale="Reds",
                title="Null Value Distribution per Column"
            )
            fig_missing.update_layout(margin=dict(l=20, r=20, t=35, b=20), height=300)
            st.plotly_chart(fig_missing, use_container_width=True)
        else:
            st.success("No missing values detected in the raw dataset!")

    st.markdown("**Column Diagnostics & Data Types**")
    st.dataframe(pd.DataFrame(initial_health["columns"]), use_container_width=True)

    with st.expander("Summary Statistics (Raw Dataset)"):
        st.dataframe(raw_df.describe(include='all').T, use_container_width=True)


# TAB 2: Cleaned Dataset & Comparison
with tab2:
    st.subheader("Cleaned & Preprocessed Dataset")
    st.dataframe(cleaned_df.head(15), use_container_width=True)

    st.markdown("---")
    st.subheader("Before vs. After Visual Comparison")

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        # Comparison of missing values
        comp_data = {
            "Stage": ["Raw Dataset", "Cleaned Dataset"],
            "Missing Values": [initial_health["total_missing_values"], final_health["total_missing_values"]],
            "Duplicate Rows": [initial_health["duplicate_rows"], final_health["duplicate_rows"]]
        }
        fig_comp = px.bar(
            pd.DataFrame(comp_data),
            x="Stage",
            y=["Missing Values", "Duplicate Rows"],
            barmode="group",
            title="Missing & Duplicate Values Before vs After",
            color_discrete_sequence=["#EF4444", "#10B981"]
        )
        fig_comp.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_comp, use_container_width=True)

    with col_c2:
        # Numeric column distribution comparison
        num_cols = raw_df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            selected_num_col = st.selectbox("Inspect Numeric Distribution Before vs After:", num_cols)
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=raw_df[selected_num_col].dropna(),
                name="Before (Raw)",
                opacity=0.6,
                marker_color="#EF4444"
            ))
            fig_dist.add_trace(go.Histogram(
                x=cleaned_df[selected_num_col].dropna() if selected_num_col in cleaned_df.columns else [],
                name="After (Cleaned)",
                opacity=0.6,
                marker_color="#10B981"
            ))
            fig_dist.update_layout(
                barmode='overlay',
                title=f"Distribution Comparison: {selected_num_col}",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_dist, use_container_width=True)


# TAB 3: Pipeline Audit Report
with tab3:
    st.subheader("Step-by-Step Pipeline Audit Trail")
    st.caption("Detailed log of every transformation applied to the dataset.")

    for i, step in enumerate(audit_log, start=1):
        with st.container():
            st.markdown(f"#### Step {i}: {step['step']}")
            st.write(f"ℹ️ **Action:** {step['details']}")
            if step['stats']:
                st.json(step['stats'])
            st.markdown("---")


# TAB 4: Download & Export
with tab4:
    st.subheader("Export Cleaned Data")
    st.write("Download the fully cleaned dataset ready for exploratory analysis, dashboards, or Machine Learning models.")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        # CSV Download
        csv_buffer = io.StringIO()
        cleaned_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue().encode('utf-8')

        st.download_button(
            label="⬇️ Download Cleaned CSV",
            data=csv_data,
            file_name=f"cleaned_{dataset_name if dataset_name.endswith('.csv') else dataset_name + '.csv'}",
            mime="text/csv",
            type="primary",
            use_container_width=True
        )

    with col_d2:
        # JSON Report Download
        report_data = {
            "dataset_name": dataset_name,
            "initial_health": initial_health,
            "final_health": final_health,
            "audit_trail": audit_log
        }
        json_report = json.dumps(report_data, indent=2, default=str)

        st.download_button(
            label="📄 Download Cleaning Audit Report (JSON)",
            data=json_report,
            file_name=f"cleaning_audit_report_{dataset_name}.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("---")
    st.markdown("### How to Use the Pipeline in Python Code")
    st.code(f"""from pipeline import DataCleaningPipeline
import pandas as pd

# Load any CSV
df = pd.read_csv('{dataset_name}')

# Initialize pipeline and execute
pipeline = DataCleaningPipeline()
clean_df, audit_trail, metrics = pipeline.run_pipeline(
    df,
    remove_dups=True,
    impute_nulls=True,
    handle_outliers=True
)

# Save cleaned data
clean_df.to_csv('cleaned_{dataset_name}', index=False)
print("Dataset successfully cleaned!")
""", language="python")
