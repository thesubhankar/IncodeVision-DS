# IncodeVision Data Science Internship Tasks 🚀

[![Task 1 App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://incodevision-task-1-datacleaningstudio.streamlit.app)
[![Task 2 App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://incodevision-task-2-eda-dashboad.streamlit.app)

This repository contains all 4 Data Science internship tasks assigned by IncodeVision, featuring automated processing pipelines, exploratory analyses, and interactive Streamlit web applications.

> 🌐 **Live Web Applications:**
> - 🧹 **Task 01 - Data Cleaning Studio:** [https://incodevision-task-1-datacleaningstudio.streamlit.app](https://incodevision-task-1-datacleaningstudio.streamlit.app)
> - 📊 **Task 02 - EDA Analytics Dashboard:** [https://incodevision-task-2-eda-dashboad.streamlit.app](https://incodevision-task-2-eda-dashboad.streamlit.app)

---

## 📌 Tasks Overview

| Task | Title | Status | Folder / Live Demo |
| :--- | :--- | :---: | :--- |
| **Task 01** | **Data Cleaning and Preprocessing** | ✅ Completed | [`Task-1(datacleaning)`](./Task-1(datacleaning)/) • [🌐 **Live App**](https://incodevision-task-1-datacleaningstudio.streamlit.app) |
| **Task 02** | **Exploratory Data Analysis (EDA)** | ✅ Completed | [`Task-2 (EDA)`](./Task-2%20(EDA)/) • [📊 **Live Dashboard**](https://incodevision-task-2-eda-dashboad.streamlit.app) |
| **Task 03** | **Sales Prediction Model** | ⏳ Upcoming | [`Task-3 (Sales prediction model)`](./Task-3%20(Sales%20prediction%20model)/) |
| **Task 04** | **Customer Segmentation Analysis** | ⏳ Upcoming | [`Task-4(costomer segmentation analysis)`](./Task-4(costomer%20segmentation%20analysis)/) |

---

## 🧹 Task 01: Data Cleaning and Preprocessing (Completed)

> 🚀 **Live Streamlit Web Application:** [**Task-1 Data Cleaning Studio**](https://incodevision-task-1-datacleaningstudio.streamlit.app)  
> *Upload any CSV or Excel dataset to automatically audit, clean missing values, eliminate duplicates, handle outliers, and export clean data in real-time.*

![Task 01 Infographic](./Task-1(datacleaning)/task-1.jpeg)

Created a complete automated pipeline, interactive Streamlit web application, and a step-by-step manual Jupyter notebook for cleaning raw tabular datasets.

- **Live Demo**: Access the interactive dashboard live on [Streamlit Cloud](https://incodevision-task-1-datacleaningstudio.streamlit.app).
- **Missing Values**: Imputed using Median (numerical) and Mode (categorical).
- **Duplicates**: Identified and removed redundant rows and duplicate IDs.
- **Format Normalization**: Standardized booleans, whitespaces, and numeric types.
- **Outliers**: Detected and treated using IQR (Interquartile Range) capping.
- **Files Included**:
  - `pipeline.py`: Reusable universal data cleaning pipeline engine.
  - `app.py`: Streamlit web app with custom file upload & live dashboard.
  - `task1_data_cleaning.ipynb`: Manual step-by-step Jupyter Notebook.
  - `cleaned_student_performance.csv`: Processed clean dataset output.
  - `run_app.bat`: 1-Click desktop launcher.

---

## 📊 Task 02: Exploratory Data Analysis (EDA) (Completed)

> 📊 **Live Streamlit Dashboard:** [**Task-2 EDA Analytics Studio**](https://incodevision-task-2-eda-dashboad.streamlit.app)  
> *Interactive analytics dashboard featuring dynamic demographic filters, statistical distributions, correlation matrices, and automated insights in real-time.*

![Task 02 Infographic](./Task-2%20(EDA)/task-2.jpeg)

Performed comprehensive Exploratory Data Analysis (EDA) and built an interactive visual analytics dashboard to discover distributions, trends, correlations, and predictive patterns.

- **Live Demo**: Access the interactive EDA dashboard live on [Streamlit Cloud](https://incodevision-task-2-eda-dashboad.streamlit.app).
- **Dashboard Features**:
  - **KPI Scorecard**: Real-time metrics for Total Records, Average Final Grade, Attendance Rate, Study Hours, and Passing Rate.
  - **Univariate Distributions**: Interactive Plotly Histograms with KDE curves, boxplots, violin plots, and categorical donut/bar charts.
  - **Bivariate & Target Analysis**: Scatter plots with OLS linear regression trendlines, correlation indicators, and demographic group-by tables.
  - **Multivariate & Correlation**: Interactive Pearson & Spearman Heatmaps, Predictive Strength Ranking, and 3D Interactive Feature Scatter.
  - **Automated Insights**: Dynamic identification of academic performance drivers (Parental Support multiplier, attendance thresholds).
  - **Export Studio**: Download filtered cohort datasets (CSV) and full EDA summary reports (JSON).
- **Files Included**:
  - `app.py`: Interactive Streamlit Exploratory Data Analysis Dashboard.
  - `run_app.bat`: 1-Click desktop launcher for the EDA Dashboard.
  - `task2_eda_analysis.ipynb`: Fully executed Jupyter Notebook with embedded charts and narrative findings.
  - `student_performance.csv`: Dataset used for analysis.
  - `task-2.jpeg`: High-resolution EDA workflow infographic poster.

---

## 📈 Task 03: Sales Prediction Model
*Upcoming — Building regression models (Linear Regression, Decision Tree, Random Forest) to predict sales.*

---

## 👥 Task 04: Customer Segmentation Analysis
*Upcoming — Applying clustering algorithms (K-Means / Hierarchical) to group customers based on behavior & spending.*
