@echo off
title Running Task-02 EDA Dashboard
cd /d "%~dp0"
echo ========================================================
echo Starting Task-02: Exploratory Data Analysis (EDA) Studio...
echo ========================================================
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Something went wrong. Press any key to exit...
    pause
)
