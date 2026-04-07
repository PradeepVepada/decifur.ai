@echo off
REM HybdRAG Startup Script
REM Starts the Streamlit frontend (which loads RAGEngine directly)

echo ========================================
echo HybdRAG - Devreotes Research Explorer
echo ========================================
echo.

REM Check if Neo4j is running
echo Checking Neo4j connection...
.\.venv\Scripts\python.exe -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('neo4j://127.0.0.1:7687', auth=('neo4j','12345678')); d.verify_connectivity(); d.close(); print('Neo4j connected!')" 2>nul
if errorlevel 1 (
    echo ERROR: Neo4j is not running. Please start Neo4j Desktop first.
    pause
    exit /b 1
)

REM Start Streamlit UI
echo Starting Streamlit UI...
start "HybdRAG Streamlit" cmd /k "cd /d %~dp0 && .\.venv\Scripts\python.exe -m streamlit run UI\streamlit_app.py --server.port 8501"

echo.
echo ========================================
echo Service starting at: http://localhost:8501
echo ========================================
echo.
echo Press any key to stop all services...
pause

REM Kill processes
taskkill /fi "WindowTitle eq HybdRAG Streamlit*" /f 2>nul
echo Services stopped.