# IncodeVision Data Science Internship Tasks 🚀

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://incodevision-ds-pyxvzgok7nqpu2sdvq2hiy.streamlit.app/)

This repository contains all 4 Data Science internship tasks assigned by IncodeVision.

> 🌐 **Live Data Cleaning Dashboard:** Experience the interactive data cleaning & preprocessing pipeline live on Streamlit Cloud:  
> 👉 [**Launch Data Cleaning Dashboard (Live App)**](https://incodevision-ds-pyxvzgok7nqpu2sdvq2hiy.streamlit.app/)

---

## 📌 Tasks Overview

| Task | Title | Status | Folder / Live Demo |
| :--- | :--- | :---: | :--- |
| **Task 01** | **Data Cleaning and Preprocessing** | ✅ Completed | [`Task-1(datacleaning)`](./Task-1(datacleaning)/) • [🌐 **Data Cleaning Dashboard (Live)**](https://incodevision-ds-pyxvzgok7nqpu2sdvq2hiy.streamlit.app/) |
| **Task 02** | **Exploratory Data Analysis (EDA)** | ✅ Completed | [`Task-2 (EDA)`](./Task-2%20(EDA)/) |
| **Task 03** | **Sales Prediction Model** | ⏳ Upcoming | [`Task-3 (Sales prediction model)`](./Task-3%20(Sales%20prediction%20model)/) |
| **Task 04** | **Customer Segmentation Analysis** | ⏳ Upcoming | [`Task-4(costomer segmentation analysis)`](./Task-4(costomer%20segmentation%20analysis)/) |

---

## 🧹 Task 01: Data Cleaning and Preprocessing (Completed)

> 🚀 **Interactive Live Web App:** [**Data Cleaning Dashboard (Streamlit Cloud)**](https://incodevision-ds-pyxvzgok7nqpu2sdvq2hiy.streamlit.app/)  
> *Upload any CSV or Excel dataset to automatically audit, clean missing values, eliminate duplicates, handle outliers, and export clean data in real-time.*

![Task 01 Infographic](./Task-1(datacleaning)/task-1.jpeg)

Created a complete automated pipeline, interactive Streamlit web application, and a step-by-step manual Jupyter notebook for cleaning raw tabular datasets.

- **Data Cleaning Dashboard (Live)**: Access the interactive web app live on [Streamlit Cloud](https://incodevision-ds-pyxvzgok7nqpu2sdvq2hiy.streamlit.app/).
- **Missing Values**: Imputed using Median (numerical) and Mode (categorical).
- **Duplicates**: Identified and removed redundant rows and duplicate IDs.
- **Format Normalization**: Standardized booleans, whitespaces, and numeric types.
- **Outliers**: Detected and treated using IQR (Interquartile Range) capping.
- **Files Included**:
  - `pipeline.py`: Reusable universal data cleaning pipeline engine.
  - `app.py`: Streamlit web app with custom file upload & live dashboard.
  - `task1_data_cleaning.ipynb`: Manual step-by-step Jupyter Notebook.
  - `cleaned_student_performance.csv`: Processed clean dataset output.

---

## 📊 Task 02: Exploratory Data Analysis (EDA) (Completed)

![Task 02 Infographic](./Task-2%20(EDA)/task-2.jpeg)

Performed comprehensive Exploratory Data Analysis (EDA) to discover distributions, trends, correlations, and predictive patterns.

- **Statistical Profiling**: Computed Mean, Median, Standard Deviation, Skewness, and Kurtosis across all metrics.
- **Univariate Analysis**: Histograms with KDE curves and categorical distribution bar plots.
- **Bivariate Analysis**: Scatter plots with regression trendlines, boxplots & violin plots across parental support levels and gender.
- **Multivariate Analysis**: Annotated Pearson & Spearman Correlation Heatmap matrices and multi-feature pairplot grids.
- **Key Insights**: Identified `PreviousGrade`, `AttendanceRate`, and `ParentalSupport` as the primary drivers of student academic performance.
- **Files Included**:
  - `task2_eda_analysis.ipynb`: Fully executed Jupyter Notebook with embedded charts and narrative findings.
  - `student_performance.csv`: Cleaned dataset used for analysis.
  - `task-2.jpeg`: High-resolution EDA workflow infographic poster.

---

## 📈 Task 03: Sales Prediction Model
*Upcoming — Building regression models (Linear Regression, Decision Tree, Random Forest) to predict sales.*

---

## 👥 Task 04: Customer Segmentation Analysis
*Upcoming — Applying clustering algorithms (K-Means / Hierarchical) to group customers based on behavior & spending.*
