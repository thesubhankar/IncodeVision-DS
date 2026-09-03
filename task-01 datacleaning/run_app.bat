@echo off
title Running Task-01 Streamlit App
cd /d "%~dp0"
echo ========================================================
echo Starting Task-01: Data Cleaning and Preprocessing App...
echo ========================================================
python -m streamlit run app.py
if errorlevel 1 (
    echo.
    echo Something went wrong. Press any key to exit...
    pause
)
