@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM ============================================================
REM  Atelier YOLO - Lancement automatique
REM
REM  - Cree (si besoin) un env conda "yolo_atelier"
REM  - Installe les dependances depuis requirements.txt
REM  - Lance Streamlit (port 8501) et Jupyter Notebook
REM
REM  Usage : double-clic sur ce fichier.
REM ============================================================

cd /d "%~dp0"

set "ENV_NAME=yolo_atelier"
set "PY_VERSION=3.11"

echo.
echo ============================================================
echo   Atelier YOLO - Configuration et lancement
echo ============================================================
echo.

REM --- Verification : on est bien dans le bon dossier ---
if not exist "app.py" (
    echo [ERREUR] app.py introuvable dans %CD%
    echo Placez ce script dans le meme dossier que app.py.
    pause
    exit /b 1
)

REM --- Detection d'Anaconda / Miniconda ---
set "CONDA_ROOT="

REM 1) via "where conda"
for /f "delims=" %%i in ('where conda 2^>nul') do (
    set "CONDA_CMD=%%i"
    for %%a in ("!CONDA_CMD!\..\..") do set "CONDA_ROOT=%%~fa"
    if exist "!CONDA_ROOT!\Scripts\activate.bat" goto :conda_ok
    set "CONDA_ROOT="
)

REM 2) chemins courants
for %%R in (
    "%USERPROFILE%\anaconda3"
    "%USERPROFILE%\Anaconda3"
    "%USERPROFILE%\miniconda3"
    "%USERPROFILE%\Miniconda3"
    "%LOCALAPPDATA%\anaconda3"
    "%LOCALAPPDATA%\Anaconda3"
    "%LOCALAPPDATA%\miniconda3"
    "%ProgramData%\Anaconda3"
    "%ProgramData%\anaconda3"
    "%ProgramData%\Miniconda3"
    "%ProgramData%\miniconda3"
    "C:\Anaconda3"
    "C:\anaconda3"
    "C:\Miniconda3"
    "C:\miniconda3"
) do (
    if exist "%%~R\Scripts\activate.bat" (
        set "CONDA_ROOT=%%~R"
        goto :conda_ok
    )
)

echo [ERREUR] Anaconda / Miniconda introuvable.
echo.
echo Solutions :
echo   1. Ouvrir une "Anaconda Prompt" depuis le menu Demarrer,
echo      puis taper :  cd /d "%~dp0"  ^&^&  %~nx0
echo   2. Reinstaller Anaconda en cochant "Add to PATH"
echo   3. Verifier l'emplacement d'installation d'Anaconda
echo.
pause
exit /b 1

:conda_ok
echo [OK] Anaconda detecte : %CONDA_ROOT%
echo.

set "ACTIVATE=%CONDA_ROOT%\Scripts\activate.bat"
set "CONDA_EXE=%CONDA_ROOT%\Scripts\conda.exe"

REM --- Existence de l'environnement ? ---
"%CONDA_EXE%" env list 2>nul > "%TEMP%\_yolo_envlist.txt"
findstr /B /C:"%ENV_NAME% " "%TEMP%\_yolo_envlist.txt" >nul 2>&1
set "ENV_EXISTS=%ERRORLEVEL%"
del "%TEMP%\_yolo_envlist.txt" >nul 2>&1

if "%ENV_EXISTS%"=="0" (
    echo [INFO] Environnement "%ENV_NAME%" deja present.
    echo.
) else (
    echo [INFO] Creation de l'environnement "%ENV_NAME%" avec Python %PY_VERSION%
    echo        Cela peut prendre 1 a 3 minutes...
    echo.
    "%CONDA_EXE%" create -n %ENV_NAME% python=%PY_VERSION% -y
    if errorlevel 1 (
        echo.
        echo [ERREUR] Creation de l'environnement echouee.
        pause
        exit /b 1
    )
    echo.
)

REM --- Activation ---
call "%ACTIVATE%" %ENV_NAME%
if errorlevel 1 (
    echo [ERREUR] Activation de l'environnement echouee.
    pause
    exit /b 1
)

REM --- Installation des dependances ---
echo [INFO] Verification / installation des dependances Python...
echo        (premier lancement : 2 a 5 minutes ; sinon quelques secondes)
echo.
python -m pip install --upgrade pip --quiet
python -m pip install --quiet -r requirements.txt
if errorlevel 1 (
    echo.
    echo [ERREUR] Installation des dependances echouee.
    echo Verifiez votre connexion internet et le fichier requirements.txt.
    pause
    exit /b 1
)
echo [OK] Dependances pretes.
echo.

REM --- Lancement de Streamlit ---
echo [INFO] Lancement de Streamlit...
echo        (s'ouvrira automatiquement a http://localhost:8501)
start "Streamlit - Atelier YOLO" cmd /k "streamlit run app.py"

REM Petite pause pour eviter d'embouteiller les logs
timeout /t 4 /nobreak >nul

REM --- Lancement de Jupyter Notebook ---
echo [INFO] Lancement de Jupyter Notebook...
echo        (s'ouvrira automatiquement avec yolo_atelier_fr.ipynb)
start "Jupyter - Atelier YOLO" cmd /k "jupyter notebook yolo_atelier_fr.ipynb"

echo.
echo ============================================================
echo   Tout est lance !
echo ============================================================
echo.
echo   Streamlit : http://localhost:8501
echo   Jupyter   : voir la fenetre Jupyter (URL avec token)
echo.
echo   Pour ARRETER :
echo     - fermer la fenetre "Streamlit - Atelier YOLO"
echo     - fermer la fenetre "Jupyter - Atelier YOLO"
echo.
echo ============================================================
echo.
pause
endlocal
