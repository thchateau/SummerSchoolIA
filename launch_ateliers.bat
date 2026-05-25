@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

REM ============================================================
REM  Summer School IA - Lancement unifie des ateliers
REM
REM  - Cree (si besoin) un env conda "summer_school"
REM  - Installe les dependances communes Raman + YOLO
REM  - Lance l'app Streamlit unifiee app_ateliers.py
REM
REM  Les boutons "Ouvrir le notebook" dans l'app Streamlit
REM  lancent automatiquement le notebook de l'atelier choisi.
REM
REM  Usage : double-clic sur ce fichier.
REM ============================================================

cd /d "%~dp0"

set "ENV_NAME=summer_school"
set "PY_VERSION=3.11"
set "APP_FILE=app_ateliers.py"

echo.
echo ============================================================
echo   Summer School IA - Configuration et lancement
echo ============================================================
echo.

REM --- Verification : on est bien dans le bon dossier ---
if not exist "%APP_FILE%" (
    echo [ERREUR] %APP_FILE% introuvable dans %CD%
    echo Placez ce script dans le meme dossier que %APP_FILE%.
    pause
    exit /b 1
)

if not exist "requirements.txt" (
    echo [ERREUR] requirements.txt introuvable dans %CD%
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
    echo        Cela peut prendre 2 a 5 minutes...
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

REM --- Installation des dependances unifiees ---
echo [INFO] Verification / installation des dependances Python...
echo        (premier lancement : 3 a 8 minutes ; sinon quelques secondes)
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
python -m ipykernel install --user --name %ENV_NAME% --display-name "Python (Summer School)" >nul 2>&1

echo [OK] Dependances pretes.
echo.

REM --- Lancement de l'app Streamlit unifiee ---
echo [INFO] Lancement de l'application Streamlit unifiee...
echo        (s'ouvrira automatiquement a http://localhost:8501)
echo.
echo ============================================================
echo   L'application va s'ouvrir dans votre navigateur.
echo.
echo   Choisissez un atelier dans la barre laterale :
echo     - Atelier Raman
echo     - Atelier YOLO
echo.
echo   Chaque atelier propose un bouton pour ouvrir son notebook
echo   Jupyter associe dans une nouvelle fenetre.
echo ============================================================
echo.

streamlit run %APP_FILE%

echo.
echo [INFO] Application fermee.
pause
endlocal
