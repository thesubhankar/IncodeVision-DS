# Incode Vision Internship / Data Science Tasks 🚀

Welcome to the project repository. This repository contains all practical tasks and machine learning pipelines developed for the internship.

---

## 📂 Repository Directory Structure

```text
d:\19.1- incode vision/
│
├── README.md                           # Main Root Documentation (This file)
│
├── task image/                         # Task instruction slides & problem statements
│   ├── task-1.jpeg
│   ├── task-2.jpeg
│   ├── task-3.jpeg
│   └── task-4.jpeg
│
├── task-01 datacleaning/               # 🧹 Task 01: Data Cleaning & Preprocessing Pipeline
│   ├── app.py                          # Streamlit Live Web Application (Interactive UI)
│   ├── pipeline.py                     # Universal Data Cleaning & Preprocessing Engine
│   ├── task1_data_cleaning.ipynb       # Manual Step-by-Step Jupyter Notebook (with outputs & plots)
│   ├── student_performance_updated_1000.csv # Raw Dataset
│   ├── cleaned_student_performance.csv # Final Processed & Cleaned Dataset
│   ├── run_app.bat                     # 1-Click Double-Click Launcher for Streamlit
│   └── requirements.txt                # Dependencies for Task 01
│
├── task-02 .../                        # (Upcoming Project Folder)
├── task-03 .../                        # (Upcoming Project Folder)
└── task-04 .../                        # (Upcoming Project Folder)
```

---

## 🧹 Task 01: Data Cleaning & Preprocessing Pipeline

### Overview
A production-ready data cleaning pipeline and interactive Streamlit web application capable of cleaning **any tabular CSV or Excel dataset**.

### Key Features
- **Missing Value Handling**: Imputes numeric columns using Median/Mean and categorical columns using Mode.
- **Duplicate Removal**: Identifies and eliminates redundant records & duplicate IDs.
- **Format Normalization**: Trims leading/trailing whitespaces, standardizes boolean values (`TRUE`/`FALSE`), and cleans data types.
- **Outlier Treatment**: Detects and winsorizes (caps) extreme values using the IQR (Interquartile Range) method.
- **ML Preprocessing**: Label / One-Hot encoding for categories and StandardScaler / MinMaxScaler for numerical features.
- **Visual Analytics**: Interactive Plotly bar charts, histograms, Before vs. After comparison cards, and downloadable cleaned CSV + JSON audit report.

---

### 🚀 How to Run Task 01

#### Option 1: 1-Click Launcher (Easiest)
Navigate to the `task-01 datacleaning` folder in Windows File Explorer and **double-click**:
👉 `run_app.bat`

#### Option 2: Run via Terminal / PowerShell
```powershell
cd "d:\19.1- incode vision\task-01 datacleaning"
python -m streamlit run app.py
```
Open your browser at: **[http://localhost:8501](http://localhost:8501)**

#### Option 3: Run Manual Step-by-Step Jupyter Notebook
```powershell
cd "d:\19.1- incode vision\task-01 datacleaning"
jupyter notebook task1_data_cleaning.ipynb
```

#### Option 4: Use Pipeline in Python Script
```python
from pipeline import DataCleaningPipeline
import pandas as pd

# Load any dataset
df = pd.read_csv("your_dataset.csv")

# Initialize and execute pipeline
pipeline = DataCleaningPipeline()
clean_df, audit_log, summary = pipeline.run_pipeline(
    df,
    remove_dups=True,
    impute_nulls=True,
    handle_outliers=True
)

# Export clean data
clean_df.to_csv("cleaned_data.csv", index=False)
```

---

## 📋 Upcoming Tasks Roadmap

- **Task 02**: Exploratory Data Analysis & Feature Engineering
- **Task 03**: Machine Learning Model Training & Evaluation
- **Task 04**: Model Deployment & Monitoring
