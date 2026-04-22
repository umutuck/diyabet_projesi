@echo off
setlocal

cd /d "%~dp0"

REM Use project virtual environment if available
if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

REM Always train the model
echo Training model...
"%PY%" model.py
if errorlevel 1 (
    echo model.py failed. Press any key to exit.
    pause >nul
    exit /b 1
)

REM Ensure streamlit is installed in selected Python
"%PY%" -m pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo streamlit not found. Installing dependencies...
    "%PY%" -m pip install streamlit numpy pandas scikit-learn matplotlib seaborn xgboost
    if errorlevel 1 (
        echo Dependency installation failed. Press any key to exit.
        pause >nul
        exit /b 1
    )
)

echo Starting app...
"%PY%" -m streamlit run app.py

endlocal
