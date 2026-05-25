@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM ============================================================
REM  Atelier Raman - Lancement automatique
REM
REM  - Cree (si besoin) un env conda "raman_atelier"
REM  - Installe les dependances depuis requirements.txt
REM  - Lance Streamlit (port 8501) et Jupyter Notebook
REM
REM  Usage : double-clic sur ce fichier.
REM ============================================================

cd /d "%~dp0"

set "ENV_NAME=raman_atelier"
set "PY_VERSION=3.11"
set "STREAMLIT_FILE=app_raman_classification.py"
set "NOTEBOOK_FILE=Raman_Classification_Binaire.ipynb"

echo.
echo ============================================================
echo   Atelier Raman - Configuration et lancement
echo ============================================================
echo.

REM --- Verification : on est bien dans le bon dossier ---
if not exist "%STREAMLIT_FILE%" (
    echo [ERREUR] %STREAMLIT_FILE% introuvable dans %CD%
    echo Placez ce script dans le meme dossier que %STREAMLIT_FILE%.
    pause
    exit /b 1
)

if not exist "%NOTEBOOK_FILE%" (
    echo [ERREUR] %NOTEBOOK_FILE% introuvable dans %CD%
    echo Placez ce script dans le meme dossier que %NOTEBOOK_FILE%.
    pause
    exit /b 1
)

REM --- Verifier que conda est disponible ---
where conda >nul 2>&1
if errorlevel 1 (
    echo [ERREUR] La commande "conda" est introuvable.
    echo.
    echo   Solutions possibles :
    echo     - Lancez ce script depuis "Anaconda Prompt"
    echo       au lieu d'un cmd / PowerShell classique.
    echo     - Ou ajoutez Anaconda au PATH systeme.
    echo.
    pause
    exit /b 1
)

REM --- Existence de l'environnement ? ---
call conda env list | findstr /R /C:"^%ENV_NAME% " >nul
if errorlevel 1 (
    echo [INFO] Creation de l'environnement "%ENV_NAME%" avec Python %PY_VERSION%
    echo        Cela peut prendre 1 a 3 minutes...
    echo.
    call conda create -n %ENV_NAME% python=%PY_VERSION% -y
    if errorlevel 1 (
        echo.
        echo [ERREUR] Creation de l'environnement echouee.
        pause
        exit /b 1
    )
    echo.
) else (
    echo [INFO] Environnement "%ENV_NAME%" deja present.
    echo.
)

REM --- Activation ---
call conda activate %ENV_NAME%
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

REM --- Enregistrement du kernel Jupyter ---
python -m ipykernel install --user --name %ENV_NAME% --display-name "Python (Raman Atelier)" >nul 2>&1

echo [OK] Dependances pretes.
echo.

REM --- Lancement de Streamlit ---
echo [INFO] Lancement de Streamlit...
echo        (s'ouvrira automatiquement a http://localhost:8501)
start "Streamlit - Atelier Raman" cmd /k "call conda activate %ENV_NAME% && cd /d "%~dp0" && streamlit run %STREAMLIT_FILE%"

REM Petite pause pour eviter d'embouteiller les logs
timeout /t 4 /nobreak >nul

REM --- Lancement de Jupyter Notebook ---
echo [INFO] Lancement de Jupyter Notebook...
echo        (s'ouvrira automatiquement avec %NOTEBOOK_FILE%)
start "Jupyter - Atelier Raman" cmd /k "call conda activate %ENV_NAME% && cd /d "%~dp0" && jupyter notebook %NOTEBOOK_FILE%"

echo.
echo ============================================================
echo   Tout est lance !
echo ============================================================
echo.
echo   Streamlit : http://localhost:8501
echo   Jupyter   : voir la fenetre Jupyter (URL avec token)
echo.
echo   Pour ARRETER :
echo     - fermer la fenetre "Streamlit - Atelier Raman"
echo     - fermer la fenetre "Jupyter - Atelier Raman"
echo.
echo ============================================================
echo.
pause
endlocal
