@echo off
setlocal
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo ========================================
echo HybdRAG — Next.js UI + FastAPI
echo ========================================
echo.

if exist ".venv\Scripts\python.exe" (
  set "PY=.venv\Scripts\python.exe"
) else (
  set "PY=python"
)

echo Checking Neo4j (optional)...
"%PY%" -c "from neo4j import GraphDatabase; d=GraphDatabase.driver('neo4j://127.0.0.1:7687', auth=('neo4j','12345678')); d.verify_connectivity(); d.close(); print('Neo4j OK')" 2>nul
if errorlevel 1 (
  echo WARNING: Neo4j not reachable. Start Neo4j Desktop if the API fails on startup.
  echo.
)

where npm >nul 2>nul
if errorlevel 1 (
  echo ERROR: npm not found. Install Node.js LTS and retry.
  pause
  exit /b 1
)

if not exist "UI\node_modules\" (
  echo Installing UI dependencies ^(UI\^)...
  call npm run install:ui
  if errorlevel 1 (
    echo npm install failed.
    pause
    exit /b 1
  )
)

echo Starting FastAPI (port 8000)...
start "HybdRAG API" cmd /k "cd /d "%ROOT%" && "%PY%" api.py"

timeout /t 4 /nobreak >nul

echo Starting Next.js dev server (cwd=UI — required for Tailwind/CSS)...
echo If port 3000 is stuck, Next uses 3001+ — use the URL printed in the Web UI window.
echo Unstyled page usually means an old process still holds :3000 — close stray Node windows or Task Manager ^> End "node.exe" on 3000.
start "HybdRAG Web UI" cmd /k "cd /d ""%ROOT%UI"" && npm run dev"

echo.
echo  API docs:  http://127.0.0.1:8000/docs
echo  Web UI:   http://127.0.0.1:3000
echo  Sign in: use any email/password (local dev stub).
echo.
echo Close each window to stop that service. Press any key to exit this launcher.
pause >nul
