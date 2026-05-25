"""
Application Streamlit unifiee Summer School IA.

Regroupe les deux ateliers etudiants dans une seule interface :
  - Atelier Raman : Classification binaire de spectres Raman (PyTorch)
  - Atelier YOLO  : Fine-tuning d'un detecteur d'objets YOLO11

Chaque atelier est execute dans son dossier d'origine (students_ramen/
ou students_yolo/) afin de preserver l'acces a ses donnees, sans
modifier les scripts existants.

Lancement :
    streamlit run app_ateliers.py
"""

from __future__ import annotations

import os
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------------------------
# Configuration de la page (doit etre le tout premier appel Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Ateliers Summer School IA",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
RAMAN_DIR = ROOT / "students_ramen"
RAMAN_APP = RAMAN_DIR / "app_raman_classification.py"
RAMAN_NOTEBOOK = RAMAN_DIR / "Raman_Classification_Binaire.ipynb"
YOLO_DIR = ROOT / "students_yolo"
YOLO_APP = YOLO_DIR / "app.py"
YOLO_NOTEBOOK = YOLO_DIR / "yolo_atelier_fr.ipynb"


# ---------------------------------------------------------------------------
# Page d'accueil
# ---------------------------------------------------------------------------
def show_landing() -> None:
    st.title("🎓 Summer School IA — Ateliers")
    st.markdown(
        """
        Bienvenue ! Cette application regroupe les **deux ateliers** de la
        Summer School en une seule interface.

        Choisissez un atelier dans la **barre laterale** a gauche.
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔬 Atelier Raman")
        st.markdown(
            """
            **Classification binaire de spectres Raman**

            - Detection cancer / normal a partir de spectres Raman
            - Reseau de neurones fully-connected (PyTorch)
            - Exploration du dataset, entrainement interactif,
              metriques (accuracy, ROC-AUC, matrice de confusion)
            """
        )
        st.caption(f"Source : `students_ramen/app_raman_classification.py`")

    with col2:
        st.subheader("🎯 Atelier YOLO")
        st.markdown(
            """
            **Fine-tuning d'un detecteur d'objets**

            - Initiation a la detection d'objets et aux CNN
            - Generation d'un mini-dataset synthetique
            - Fine-tuning de YOLO11n sur CPU
            - Metriques (IoU, precision, rappel, mAP) et inference
            """
        )
        st.caption(f"Source : `students_yolo/app.py`")

    st.markdown("---")
    st.info(
        "ℹ️ Chaque atelier conserve ses propres donnees et son propre "
        "environnement de travail. La barre laterale affiche les options "
        "specifiques a l'atelier selectionne."
    )


# ---------------------------------------------------------------------------
# Lancement d'un notebook Jupyter associe a l'atelier
# ---------------------------------------------------------------------------
def launch_notebook(notebook_path: Path) -> None:
    """Ouvre le notebook dans une instance Jupyter Notebook locale.

    Lance `jupyter notebook <notebook>` en sous-processus detache, dans le
    dossier du notebook, ce qui ouvre une nouvelle fenetre/onglet de
    navigateur sur l'URL Jupyter (port 8888 par defaut).
    """
    if not notebook_path.exists():
        st.sidebar.error(f"Notebook introuvable : `{notebook_path.name}`")
        return

    if shutil.which("jupyter") is None:
        st.sidebar.error(
            "Commande `jupyter` introuvable dans l'environnement courant. "
            "Verifiez que jupyter / notebook est installe."
        )
        return

    try:
        subprocess.Popen(
            ["jupyter", "notebook", notebook_path.name],
            cwd=str(notebook_path.parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        st.sidebar.success(
            f"Notebook lance : `{notebook_path.name}`\n\n"
            "Une nouvelle fenetre de navigateur va s'ouvrir "
            "(http://localhost:8888)."
        )
    except OSError as exc:
        st.sidebar.error(f"Echec du lancement de Jupyter : {exc}")


# ---------------------------------------------------------------------------
# Delegation vers un script Streamlit existant
# ---------------------------------------------------------------------------
def run_external_app(app_path: Path) -> None:
    """Execute un script Streamlit existant dans son dossier d'origine.

    - Change le repertoire courant pour que les chemins relatifs (par ex.
      le dossier `Ramen/` du dataset Raman) soient correctement resolus.
    - Ajoute le dossier de l'app au `sys.path` pour les imports relatifs.
    - Neutralise temporairement `st.set_page_config` : un seul appel par
      run est autorise par Streamlit, on l'a deja fait au-dessus.
    """
    app_dir = app_path.parent

    original_set_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None  # type: ignore[assignment]

    previous_cwd = os.getcwd()
    added_to_path = False
    try:
        os.chdir(app_dir)
        if str(app_dir) not in sys.path:
            sys.path.insert(0, str(app_dir))
            added_to_path = True
        runpy.run_path(str(app_path), run_name="__main__")
    finally:
        os.chdir(previous_cwd)
        if added_to_path:
            try:
                sys.path.remove(str(app_dir))
            except ValueError:
                pass
        st.set_page_config = original_set_page_config  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🎓 Ateliers")
choice = st.sidebar.selectbox(
    "Atelier",
    ["🏠 Accueil", "🔬 Atelier Raman", "🎯 Atelier YOLO"],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")

if choice == "🏠 Accueil":
    show_landing()
elif choice == "🔬 Atelier Raman":
    if st.sidebar.button("📓 Ouvrir le notebook Raman", use_container_width=True):
        launch_notebook(RAMAN_NOTEBOOK)
    st.sidebar.markdown("---")
    if not RAMAN_APP.exists():
        st.error(f"Fichier introuvable : `{RAMAN_APP}`")
    else:
        run_external_app(RAMAN_APP)
else:  # YOLO
    if st.sidebar.button("📓 Ouvrir le notebook YOLO", use_container_width=True):
        launch_notebook(YOLO_NOTEBOOK)
    st.sidebar.markdown("---")
    if not YOLO_APP.exists():
        st.error(f"Fichier introuvable : `{YOLO_APP}`")
    else:
        run_external_app(YOLO_APP)
