"""
Application Streamlit unifiee Summer School IA.

Regroupe les deux ateliers etudiants dans une seule interface :
  - Atelier Raman : Classification binaire de spectres Raman (PyTorch)
  - Atelier YOLO  : Fine-tuning d'un detecteur d'objets YOLO11

Chaque atelier est execute en deleguant a son script Streamlit d'origine,
sans le modifier. Compatible Streamlit Community Cloud.

Lancement local :
    streamlit run app_ateliers.py
"""

from __future__ import annotations

import os
import runpy
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
YOLO_DIR = ROOT / "students_yolo"
YOLO_APP = YOLO_DIR / "app.py"

# Liens Google Colab vers les notebooks Jupyter associes a chaque atelier.
# Colab ouvre directement les .ipynb hebergés sur GitHub.
GITHUB_REPO = "thchateau/SummerSchoolIA"
GITHUB_BRANCH = "main"
RAMAN_NOTEBOOK_REL = "students_ramen/Raman_Classification_Binaire.ipynb"
YOLO_NOTEBOOK_REL = "students_yolo/yolo_atelier_fr.ipynb"


def colab_url(notebook_rel_path: str) -> str:
    return (
        f"https://colab.research.google.com/github/"
        f"{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{notebook_rel_path}"
    )


def github_url(notebook_rel_path: str) -> str:
    return f"https://github.com/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{notebook_rel_path}"


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
        st.caption("Source : `students_ramen/app_raman_classification.py`")

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
        st.caption("Source : `students_yolo/app.py`")

    st.markdown("---")
    st.info(
        "ℹ️ Chaque atelier propose dans sa barre laterale un bouton pour "
        "**ouvrir le notebook Jupyter associe dans Google Colab** "
        "(execution gratuite en ligne, pas d'installation locale requise)."
    )


# ---------------------------------------------------------------------------
# Delegation vers un script Streamlit existant
# ---------------------------------------------------------------------------
def run_external_app(app_path: Path, cwd: Path | None = None) -> None:
    """Execute un script Streamlit existant en preservant son contexte.

    - `cwd` : repertoire courant a utiliser pendant l'execution.
      Pour l'app Raman, on utilise la racine du repo afin que les chemins
      relatifs (par ex. `Ramen/`) soient resolus correctement.
    - Ajoute le dossier de l'app au `sys.path` pour les imports relatifs.
    - Neutralise temporairement `st.set_page_config` : un seul appel par
      run est autorise par Streamlit, on l'a deja fait au-dessus.
    """
    app_dir = app_path.parent
    target_cwd = cwd if cwd is not None else app_dir

    original_set_page_config = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None  # type: ignore[assignment]

    previous_cwd = os.getcwd()
    added_to_path = False
    try:
        os.chdir(target_cwd)
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
    st.sidebar.link_button(
        "📓 Ouvrir le notebook (Colab)",
        colab_url(RAMAN_NOTEBOOK_REL),
        use_container_width=True,
    )
    st.sidebar.link_button(
        "👁️ Voir le notebook (GitHub)",
        github_url(RAMAN_NOTEBOOK_REL),
        use_container_width=True,
    )
    st.sidebar.markdown("---")
    if not RAMAN_APP.exists():
        st.error(f"Fichier introuvable : `{RAMAN_APP}`")
    else:
        # L'app Raman cherche `Ramen/` dans son CWD ; on la lance depuis la
        # racine du repo ou ce dossier existe deja (cf. depot SummerSchoolIA).
        run_external_app(RAMAN_APP, cwd=ROOT)

else:  # YOLO
    st.sidebar.link_button(
        "📓 Ouvrir le notebook (Colab)",
        colab_url(YOLO_NOTEBOOK_REL),
        use_container_width=True,
    )
    st.sidebar.link_button(
        "👁️ Voir le notebook (GitHub)",
        github_url(YOLO_NOTEBOOK_REL),
        use_container_width=True,
    )
    st.sidebar.markdown("---")
    if not YOLO_APP.exists():
        st.error(f"Fichier introuvable : `{YOLO_APP}`")
    else:
        run_external_app(YOLO_APP)
