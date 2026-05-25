"""
Application Streamlit pédagogique - Fine-tuning YOLO sur CPU.

Destinée à des étudiants débutants en IA, cette interface couvre :
- Le principe de YOLO et de la détection d'objets
- La création d'un jeu de données et le format des labels
- Les splits train / val / test
- Les notions d'epoch, de batch et de loss
- L'entraînement réel d'un modèle YOLOv8
- Les métriques d'évaluation (IoU, précision, rappel, mAP)
- L'inférence sur de nouvelles images

Lancement : streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Configuration globale de la page
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="YOLO - Tutoriel interactif",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "mini_dataset"
RUNS_PATH = ROOT / "yolo_finetuning"
RUN_NAME = "shape_detector_app"

CLASS_NAMES = {0: "rectangle", 1: "cercle", 2: "triangle"}
CLASS_COLORS = {0: "#FF6B6B", 1: "#4ECDC4", 2: "#FFD166"}

# Modèle pré-entraîné utilisé pour le fine-tuning.
# yolo11n est le plus petit modèle Ultralytics disponible (~2,6M paramètres),
# choisi pour minimiser le temps d'entraînement sur CPU.
PRETRAINED_MODEL = "yolo11n.pt"


# ---------------------------------------------------------------------------
# Fonctions utilitaires partagées entre les sections
# ---------------------------------------------------------------------------
def draw_shape(img: np.ndarray, shape_type: int, x_center: int, y_center: int,
               size: int, color: tuple[int, int, int]) -> tuple[float, float, float, float]:
    """Dessine une forme et renvoie la boîte englobante normalisée."""
    h, w = img.shape[:2]
    if shape_type == 0:  # Rectangle
        x1 = max(0, x_center - size // 2)
        y1 = max(0, y_center - size // 2)
        x2 = min(w, x_center + size // 2)
        y2 = min(h, y_center + size // 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        return (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h

    if shape_type == 1:  # Cercle
        cv2.circle(img, (x_center, y_center), size // 2, color, -1)
        return x_center / w, y_center / h, size / w, size / h

    # Triangle
    pts = np.array([
        [x_center, y_center - size // 2],
        [x_center - size // 2, y_center + size // 2],
        [x_center + size // 2, y_center + size // 2],
    ], np.int32)
    cv2.fillPoly(img, [pts], color)
    return x_center / w, y_center / h, size / w, size / h


def make_synthetic_image(seed: int, n_objects_range: tuple[int, int] = (1, 4),
                         img_size: int = 320) -> tuple[np.ndarray, list[tuple]]:
    """Crée une image synthétique et renvoie l'image + la liste des labels YOLO."""
    rng = np.random.default_rng(seed)
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
    labels: list[tuple] = []
    n_objects = int(rng.integers(n_objects_range[0], n_objects_range[1] + 1))
    for _ in range(n_objects):
        shape_type = int(rng.integers(0, 3))
        x_c = int(rng.integers(40, img_size - 40))
        y_c = int(rng.integers(40, img_size - 40))
        size = int(rng.integers(30, 70))
        color = tuple(int(v) for v in rng.integers(50, 200, 3))
        bbox = draw_shape(img, shape_type, x_c, y_c, size, color)
        labels.append((shape_type, *bbox))
    return img, labels


def draw_bbox_on_image(img: np.ndarray, bbox_norm: tuple[float, float, float, float],
                       color: str = "#FF6B6B", label: str | None = None) -> Image.Image:
    """Dessine une bbox normalisée (xc, yc, w, h) sur une image numpy."""
    pil_img = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(pil_img)
    W, H = pil_img.size
    xc, yc, w, h = bbox_norm
    x1 = int((xc - w / 2) * W)
    y1 = int((yc - h / 2) * H)
    x2 = int((xc + w / 2) * W)
    y2 = int((yc + h / 2) * H)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
    if label:
        draw.text((x1 + 4, y1 + 4), label, fill=color)
    return pil_img


def write_yolo_dataset(n_train: int, n_val: int, n_objects_range: tuple[int, int],
                       img_size: int = 320) -> dict:
    """Génère le jeu de données complet sur le disque et renvoie un résumé."""
    for split in ("train", "val"):
        (DATASET_PATH / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET_PATH / "labels" / split).mkdir(parents=True, exist_ok=True)

    summary = {"train": 0, "val": 0, "objects_train": 0, "objects_val": 0}
    for i in range(n_train):
        img, labels = make_synthetic_image(i, n_objects_range, img_size)
        Image.fromarray(img).save(DATASET_PATH / "images" / "train" / f"img_{i:03d}.jpg")
        with open(DATASET_PATH / "labels" / "train" / f"img_{i:03d}.txt", "w") as f:
            for c, xc, yc, w, h in labels:
                f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        summary["train"] += 1
        summary["objects_train"] += len(labels)

    for i in range(n_val):
        img, labels = make_synthetic_image(i + 1000, n_objects_range, img_size)
        Image.fromarray(img).save(DATASET_PATH / "images" / "val" / f"img_{i:03d}.jpg")
        with open(DATASET_PATH / "labels" / "val" / f"img_{i:03d}.txt", "w") as f:
            for c, xc, yc, w, h in labels:
                f.write(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")
        summary["val"] += 1
        summary["objects_val"] += len(labels)

    yaml_content = (
        f"path: {DATASET_PATH.absolute()}\n"
        f"train: images/train\n"
        f"val: images/val\n\n"
        f"names:\n"
        f"  0: rectangle\n"
        f"  1: cercle\n"
        f"  2: triangle\n"
    )
    (DATASET_PATH / "data.yaml").write_text(yaml_content)
    return summary


def compute_iou(box1: tuple[int, int, int, int],
                box2: tuple[int, int, int, int]) -> float:
    """IoU entre deux boîtes au format (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def find_latest_run() -> Path | None:
    """Renvoie le dossier d'entraînement le plus récent ou None."""
    run_dir = RUNS_PATH / RUN_NAME
    if (run_dir / "weights" / "best.pt").exists():
        return run_dir
    if RUNS_PATH.exists():
        candidates = sorted(
            [p for p in RUNS_PATH.iterdir() if (p / "weights" / "best.pt").exists()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if candidates:
            return candidates[0]
    return None


# ---------------------------------------------------------------------------
# Sidebar : navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🎯 Tutoriel YOLO")
st.sidebar.markdown(
    "Une visite guidée du fine-tuning d'un détecteur d'objets, "
    "pensée pour les débutants en IA."
)

SECTIONS = [
    "🏠 Accueil",
    "1. Qu'est-ce que la détection d'objets ?",
    "2. Les réseaux convolutifs (CNN)",
    "3. Le principe de YOLO",
    "4. Création du jeu de données",
    "5. Format des labels YOLO",
    "6. Train / Validation / Test",
    "7. Epoch, batch et loss",
    "8. Entraînement du modèle",
    "9. Métriques d'évaluation",
    "10. Inférence sur de nouvelles images",
    "11. Récapitulatif",
]

section = st.sidebar.radio("Sections", SECTIONS, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.caption(
    "💡 Astuce : suivez les sections dans l'ordre la première fois. "
    "Chaque page est interactive — n'hésitez pas à manipuler les curseurs !"
)


# ===========================================================================
# SECTION : ACCUEIL
# ===========================================================================
if section == SECTIONS[0]:
    st.title("🎯 Fine-tuning d'un détecteur d'objets YOLO")
    st.subheader("Un tutoriel interactif pour débuter en IA")

    st.markdown(
        """
        Bienvenue ! Cette application vous emmène pas à pas dans le monde de la
        **détection d'objets** avec **YOLO** (*You Only Look Once*).

        À la fin du parcours, vous saurez :
        - 🧠 ce qu'est un détecteur d'objets et comment il se distingue d'un classifieur
        - 🖼️ comment construire un **jeu de données** et écrire des **labels** au bon format
        - 📚 ce que signifient les notions d'**epoch**, de **batch** et de **loss**
        - 🚂 comment **entraîner** réellement un modèle YOLOv8 sur votre machine
        - 📊 comment **évaluer** ses performances avec les métriques standards
        - 🔮 comment l'utiliser pour **prédire** sur de nouvelles images
        """
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Sections", "10", "interactives")
    col2.metric("Concepts clés", "6+", "expliqués")
    col3.metric("Modèle utilisé", "YOLO11n", "2,6M paramètres")

    st.info(
        "📚 **Comment utiliser cette app ?** Suivez les sections dans l'ordre en "
        "utilisant la barre latérale à gauche. Chaque section contient des "
        "explications et des éléments interactifs (curseurs, boutons, etc.) "
        "pour expérimenter par vous-même."
    )

    st.markdown("### 🗺️ Plan du parcours")
    st.markdown(
        """
        | # | Section | Ce que vous y apprenez |
        |---|---------|------------------------|
        | 1 | Qu'est-ce que la détection d'objets ? | Différence classification / détection / segmentation |
        | 2 | Les réseaux convolutifs (CNN) | Convolution, filtres, pooling, hiérarchie de caractéristiques |
        | 3 | Le principe de YOLO | Pourquoi « You Only Look Once » et la grille de prédiction |
        | 4 | Création du jeu de données | Générer des images synthétiques étiquetées |
        | 5 | Format des labels YOLO | Comprendre `class xc yc w h` normalisé |
        | 6 | Train / Val / Test | Pourquoi séparer les données et l'over-fitting |
        | 7 | Epoch, batch et loss | Mécanique de l'apprentissage |
        | 8 | Entraînement du modèle | Lancer un vrai fine-tuning sur CPU |
        | 9 | Métriques | IoU, précision, rappel, mAP avec démos interactives |
        | 10 | Inférence | Détecter des objets sur de nouvelles images |
        | 11 | Récapitulatif | Synthèse et idées de prolongements |
        """
    )


# ===========================================================================
# SECTION 1 : QU'EST-CE QUE LA DÉTECTION D'OBJETS ?
# ===========================================================================
elif section == SECTIONS[1]:
    st.title("1. Qu'est-ce que la détection d'objets ?")

    st.markdown(
        """
        En vision par ordinateur, on distingue trois grandes tâches :
        """
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("#### 🏷️ Classification")
        st.markdown(
            "Le modèle répond à la question : **« qu'y a-t-il sur l'image ? »**\n\n"
            "→ Une seule étiquette pour toute l'image."
        )
        demo = np.ones((200, 200, 3), dtype=np.uint8) * 240
        cv2.circle(demo, (100, 100), 50, (78, 205, 196), -1)
        st.image(demo, caption="Résultat : « cercle »", width=200)

    with col_b:
        st.markdown("#### 🎯 Détection")
        st.markdown(
            "Le modèle répond à : **« quoi, et où ? »**\n\n"
            "→ Une étiquette **et** une boîte englobante par objet."
        )
        demo = np.ones((200, 200, 3), dtype=np.uint8) * 240
        cv2.circle(demo, (100, 100), 50, (78, 205, 196), -1)
        cv2.rectangle(demo, (50, 50), (150, 150), (255, 107, 107), 3)
        st.image(demo, caption="Résultat : « cercle » + boîte", width=200)

    with col_c:
        st.markdown("#### 🧩 Segmentation")
        st.markdown(
            "Le modèle répond pixel par pixel : **« à quel objet appartient chaque pixel ? »**\n\n"
            "→ Le contour exact de chaque objet."
        )
        demo = np.ones((200, 200, 3), dtype=np.uint8) * 240
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.circle(mask, (100, 100), 50, 255, -1)
        demo[mask > 0] = (255, 107, 107)
        st.image(demo, caption="Résultat : masque rouge", width=200)

    st.markdown("---")
    st.markdown(
        """
        ### 💡 Pourquoi la détection est-elle plus difficile que la classification ?

        Parce que le modèle doit produire **plusieurs informations** par image :
        - le **nombre** d'objets (qui varie d'une image à l'autre)
        - la **classe** de chaque objet
        - la **position** (x, y, largeur, hauteur) de chaque objet
        - un **score de confiance** pour chaque détection
        """
    )

    st.success(
        "**À retenir :** la détection d'objets, c'est répondre à *quoi* et *où* "
        "pour un nombre variable d'objets dans la même image."
    )


# ===========================================================================
# SECTION 2 : LES RÉSEAUX CONVOLUTIFS (CNN)
# ===========================================================================
elif section == SECTIONS[2]:
    st.title("2. Les réseaux convolutifs (CNN)")

    st.markdown(
        """
        Avant d'attaquer YOLO, faisons un détour par les **CNN**
        (*Convolutional Neural Networks* — réseaux de neurones convolutifs).
        C'est l'architecture qui a **révolutionné la vision par ordinateur** depuis 2012,
        et **YOLO est avant tout un grand CNN**.
        """
    )

    st.markdown("### 🤔 Pourquoi pas un réseau de neurones « classique » ?")
    st.markdown(
        """
        Une petite image 320×320 en couleur contient déjà **307 200 pixels**
        (320 × 320 × 3 canaux RVB).

        Si on connectait directement tous ces pixels à un réseau classique
        (chaque pixel relié à chaque neurone), on aurait :
        - **des millions de paramètres** dès la première couche
        - **aucune prise en compte** du fait que **deux pixels voisins sont liés**
          (un bord, une texture s'étend sur plusieurs pixels contigus)
        - **aucune invariance par translation** : le modèle apprendrait séparément
          à reconnaître un objet en haut à gauche et le même objet en bas à droite

        Les CNN règlent ces problèmes en **exploitant la structure 2D** des images.
        """
    )

    st.markdown("### 🧱 La brique de base : la convolution")
    st.markdown(
        """
        Une **convolution** consiste à faire glisser un petit **filtre** (par
        exemple une matrice 3×3) sur toute l'image. À chaque position, on
        multiplie les valeurs du filtre par les pixels correspondants et on
        additionne. Le résultat est une **carte d'activation** qui indique,
        en chaque point, *à quel point cette zone ressemble au motif* du filtre.

        Avantages :
        - **peu de paramètres** : un filtre 3×3 = seulement 9 poids, partagés
          sur toute l'image
        - **détection locale** : chaque sortie ne dépend que d'une petite
          zone de l'image
        - **invariance par translation** : le même filtre détecte le motif
          partout dans l'image
        """
    )

    st.markdown("#### 🎛️ Démo interactive : choisissez un filtre")
    st.caption(
        "Appliquez différents filtres 3×3 à une image synthétique pour voir "
        "ce que chaque type de filtre « détecte »."
    )

    col_demo_a, col_demo_b = st.columns([1, 1])
    with col_demo_a:
        kernel_choice = st.selectbox(
            "Type de filtre",
            [
                "Identité (aucun effet)",
                "Détection de contours (Laplacien)",
                "Contours verticaux (Sobel X)",
                "Contours horizontaux (Sobel Y)",
                "Flou (moyenne)",
                "Accentuation (sharpen)",
            ],
        )
        demo_seed = st.slider("Image de démo (graine aléatoire)", 0, 30, 7,
                              key="cnn_demo_seed")

    kernels = {
        "Identité (aucun effet)": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32),
        "Détection de contours (Laplacien)": np.array(
            [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
        "Contours verticaux (Sobel X)": np.array(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
        "Contours horizontaux (Sobel Y)": np.array(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
        "Flou (moyenne)": np.ones((3, 3), dtype=np.float32) / 9.0,
        "Accentuation (sharpen)": np.array(
            [[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
    }

    demo_img, _ = make_synthetic_image(demo_seed, (2, 4))
    gray = cv2.cvtColor(demo_img, cv2.COLOR_RGB2GRAY)
    kernel = kernels[kernel_choice]
    filtered = cv2.filter2D(gray, ddepth=cv2.CV_32F, kernel=kernel)
    # Normalisation pour affichage
    f_abs = np.abs(filtered)
    f_norm = (f_abs / (f_abs.max() + 1e-9) * 255).astype(np.uint8)

    with col_demo_b:
        sub_a, sub_b = st.columns(2)
        sub_a.image(demo_img, caption="Image d'origine")
        sub_b.image(f_norm, caption=f"Après convolution :\n{kernel_choice}")
        st.markdown("**Matrice du filtre 3×3** :")
        st.dataframe(pd.DataFrame(kernel), hide_index=True,
                     use_container_width=False)

    st.info(
        "👀 **Observation :** chaque filtre **« met en évidence »** un type de "
        "motif particulier. Le Laplacien fait ressortir tous les contours, "
        "Sobel X uniquement les contours verticaux, etc. Un CNN apprend "
        "automatiquement des **centaines de filtres** comme ceux-ci."
    )

    st.markdown("### 🎓 Les filtres sont **appris**, pas écrits à la main")
    st.markdown(
        """
        C'est la grande force des CNN : **on ne dit jamais au réseau quels
        filtres utiliser**. Les valeurs des filtres sont **ajustées
        automatiquement** par l'entraînement, en faisant descendre la loss.

        Résultat empirique observé dans tous les CNN visuels :
        - 🟦 **Premières couches** → filtres qui détectent des **contours**,
          des **lignes**, des **dégradés de couleur**
        - 🟨 **Couches intermédiaires** → **textures**, **coins**, **petits motifs**
        - 🟥 **Couches profondes** → **parties d'objets** (roue, œil, aile…),
          puis **objets entiers**
        """
    )

    st.markdown("### 📉 Le pooling : réduire la taille")
    st.markdown(
        """
        Entre les couches de convolution, on intercale souvent du **pooling**
        (en général **max-pooling 2×2**), qui :
        - **divise par 2** la hauteur et la largeur de la carte d'activation
        - **rend le réseau plus robuste** aux petits décalages
        - **réduit le coût** de calcul des couches suivantes
        """
    )

    pool_demo = np.array([
        [3, 1, 4, 1],
        [5, 9, 2, 6],
        [5, 3, 5, 8],
        [9, 7, 9, 3],
    ])
    pool_result = np.array([
        [max(3, 1, 5, 9), max(4, 1, 2, 6)],
        [max(5, 3, 9, 7), max(5, 8, 9, 3)],
    ])
    col_p1, col_p2, col_p3 = st.columns([2, 1, 2])
    with col_p1:
        st.markdown("**Avant pooling (4×4)**")
        st.dataframe(pd.DataFrame(pool_demo), hide_index=True)
    with col_p2:
        st.markdown("<br><br><div style='text-align:center;font-size:36px;'>→</div>",
                    unsafe_allow_html=True)
        st.caption("max-pool 2×2")
    with col_p3:
        st.markdown("**Après max-pooling (2×2)**")
        st.dataframe(pd.DataFrame(pool_result), hide_index=True)

    st.markdown("### 🏗️ L'architecture globale d'un CNN")
    st.markdown(
        """
        Un CNN typique enchaîne ces blocs **plusieurs fois** :

        ```
        Image
           ↓
        [Convolution + activation (ReLU)]   ← détecter des motifs
        [Pooling]                            ← réduire la taille
           ↓
        [Convolution + ReLU]                 ← motifs plus complexes
        [Pooling]
           ↓
        ...   (5 à 50 fois selon le réseau)
           ↓
        [Tête spécifique à la tâche]         ← classification, détection, …
        ```

        À chaque étage, la carte devient **plus petite spatialement** mais
        **plus riche sémantiquement** (plus de canaux, contenant chacun une
        caractéristique différente).
        """
    )

    st.markdown("### 🔗 Et YOLO dans tout ça ?")
    st.markdown(
        """
        YOLO est composé de **deux parties** :

        1. **Un backbone CNN** (la « colonne vertébrale »), qui extrait les
           caractéristiques hiérarchiques que nous venons de voir.
           Pour YOLO11n, ce backbone fait ~2,6 millions de paramètres.
        2. **Une tête de détection**, qui prend les cartes de caractéristiques
           du backbone et prédit les **boîtes englobantes**, les **classes**,
           et les **scores de confiance**.

        C'est exactement ce qu'on découvre dans la **section suivante** !
        """
    )

    st.success(
        "**À retenir :**\n"
        "- Un **CNN** apprend automatiquement des **filtres** à appliquer sur l'image\n"
        "- L'empilement de **convolution + pooling** crée une **hiérarchie** de "
        "caractéristiques : contours → textures → parties → objets\n"
        "- **YOLO** = un CNN (backbone) + une tête de détection spécialisée"
    )


# ===========================================================================
# SECTION 3 : LE PRINCIPE DE YOLO
# ===========================================================================
elif section == SECTIONS[3]:
    st.title("3. Le principe de YOLO")

    st.markdown(
        """
        **YOLO** signifie *You Only Look Once* — « tu ne regardes qu'une seule fois ».

        Avant YOLO, les détecteurs (comme R-CNN) procédaient en **deux étapes** :
        1. proposer plein de régions candidates dans l'image
        2. classifier chacune de ces régions séparément

        C'était lent. YOLO change le paradigme : **une seule passe** sur l'image
        suffit pour prédire **simultanément** les boîtes et les classes.
        """
    )

    st.markdown("### 🗂️ Le principe de la grille")
    st.markdown(
        """
        YOLO découpe l'image en une **grille** (par exemple 7×7 ou 13×13 selon la version).

        Pour chaque cellule de la grille, le modèle prédit :
        - 📦 plusieurs **boîtes candidates** (avec position et taille)
        - 🎯 un **score de confiance** par boîte (est-ce qu'il y a un objet ?)
        - 🏷️ une **distribution de probabilités** sur les classes possibles

        Tous ces nombres sortent **en parallèle** d'un seul passage dans le réseau.
        """
    )

    grid_size = st.slider("Taille de la grille (cellules de côté)", 3, 13, 7)
    img_size = 400
    grid_img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
    cell = img_size // grid_size
    for i in range(1, grid_size):
        cv2.line(grid_img, (i * cell, 0), (i * cell, img_size), (180, 180, 180), 1)
        cv2.line(grid_img, (0, i * cell), (img_size, i * cell), (180, 180, 180), 1)
    cv2.circle(grid_img, (img_size // 2, img_size // 2), img_size // 5,
               (78, 205, 196), -1)
    cv2.rectangle(grid_img,
                  (img_size // 2 - img_size // 5, img_size // 2 - img_size // 5),
                  (img_size // 2 + img_size // 5, img_size // 2 + img_size // 5),
                  (255, 107, 107), 3)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(grid_img, caption=f"Grille {grid_size}×{grid_size} avec un objet "
                                   f"({grid_size * grid_size} cellules)")
    with col2:
        st.metric("Cellules de la grille", grid_size * grid_size)
        st.metric("Prédictions par cellule", "≈ B boîtes × (4 coords + 1 conf) + C classes")
        st.caption(
            f"Avec {grid_size * grid_size} cellules, B=3 boîtes et C=3 classes : "
            f"chaque image génère **{grid_size * grid_size * (3 * 5 + 3)} prédictions brutes** "
            "(la plupart seront éliminées par un seuil de confiance et la NMS — "
            "Non-Maximum Suppression)."
        )

    st.markdown("### ⚡ Pourquoi YOLO est-il rapide ?")
    st.markdown(
        """
        - **Une seule passe** dans le réseau, contre des centaines pour R-CNN
        - **Architecture entièrement convolutionnelle** : très efficace sur GPU
        - **Petites variantes** disponibles (YOLOv8n = 3 M paramètres)

        | Modèle | Paramètres | Vitesse CPU |
        |--------|-----------|-------------|
        | **YOLO11n (nano)** | **2,6 M** | **⚡⚡⚡⚡ (le plus rapide)** |
        | YOLOv8n (nano) | 3,2 M | ⚡⚡⚡ |
        | YOLO11s (small) | 9,4 M | ⚡⚡ |
        | YOLO11m (medium) | 20,1 M | ⚡ |
        | YOLO11l (large) | 25,3 M | 🐌 |
        | YOLO11x (xlarge) | 56,9 M | 🐌🐌 |

        Dans ce tutoriel, on utilise **YOLO11n** — le plus petit modèle YOLO disponible
        chez Ultralytics (~2,6 M de paramètres), idéal pour un entraînement rapide sur CPU.
        """
    )

    st.success(
        "**À retenir :** YOLO traite toute l'image en une seule passe et prédit "
        "en parallèle position + classe + confiance pour de nombreuses boîtes candidates."
    )


# ===========================================================================
# SECTION 4 : CRÉATION DU JEU DE DONNÉES
# ===========================================================================
elif section == SECTIONS[4]:
    st.title("4. Création du jeu de données")

    st.markdown(
        """
        Un modèle de détection apprend à partir **d'exemples** : des images
        accompagnées de leurs **annotations** (les boîtes englobantes avec leur classe).

        Pour ce tutoriel, on triche un peu : au lieu de collecter et d'annoter
        des photos à la main, on **génère** des images synthétiques avec des formes
        géométriques. Avantage : on a immédiatement les labels exacts (puisqu'on
        sait où on a dessiné chaque forme).

        ### 🗂️ Structure de dossier attendue par YOLO

        ```
        mini_dataset/
        ├── images/
        │   ├── train/   ← images d'entraînement
        │   └── val/     ← images de validation
        ├── labels/
        │   ├── train/   ← un .txt par image, même nom
        │   └── val/
        └── data.yaml    ← fichier de configuration
        ```

        Chaque image `img_042.jpg` a un fichier de labels `img_042.txt` du même nom.
        """
    )

    st.markdown("### 🎛️ Générez votre propre jeu de données")
    st.caption(
        "Réglez les paramètres puis cliquez sur le bouton pour générer un dataset complet sur le disque."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        n_train = st.slider("Images d'entraînement", 10, 100, 40, step=5)
    with col2:
        n_val = st.slider("Images de validation", 5, 30, 10, step=1)
    with col3:
        objects_range = st.slider(
            "Objets par image (min – max)", 1, 6, (1, 3),
            help="Plus il y a d'objets, plus la tâche est difficile.",
        )

    if st.button("🛠️ Générer le jeu de données", type="primary"):
        with st.spinner("Génération en cours..."):
            summary = write_yolo_dataset(n_train, n_val, tuple(objects_range))
        st.success(
            f"✅ Dataset généré dans `{DATASET_PATH}` : "
            f"{summary['train']} images d'entraînement "
            f"({summary['objects_train']} objets) et "
            f"{summary['val']} images de validation "
            f"({summary['objects_val']} objets)."
        )

    st.markdown("### 👁️ Aperçu de quelques images")
    preview_seed = st.slider("Graine aléatoire pour l'aperçu", 0, 50, 0)
    cols = st.columns(4)
    for i, col in enumerate(cols):
        img, labels = make_synthetic_image(preview_seed + i, tuple(objects_range))
        col.image(img, caption=f"Image {preview_seed + i} — {len(labels)} objet(s)")

    st.info(
        "🎓 **En projet réel**, vous remplacerez cette génération par :\n"
        "1. la collecte de **vraies photos** (téléphone, internet, capteurs...)\n"
        "2. l'**annotation manuelle** avec un outil comme [LabelImg](https://github.com/tzutalin/labelImg), "
        "[Roboflow](https://roboflow.com/) ou [CVAT](https://www.cvat.ai/)\n"
        "3. l'**export au format YOLO** (le même format que ce que vous voyez ici)"
    )


# ===========================================================================
# SECTION 5 : FORMAT DES LABELS YOLO
# ===========================================================================
elif section == SECTIONS[5]:
    st.title("5. Format des labels YOLO")

    st.markdown(
        """
        Pour chaque image, YOLO attend un fichier `.txt` du même nom (par exemple
        `img_042.jpg` → `img_042.txt`). Ce fichier contient **une ligne par objet** :

        ```
        class_id x_center y_center width height
        ```

        Les 4 nombres de coordonnées sont **normalisés entre 0 et 1**, c'est-à-dire
        exprimés comme une **fraction** de la taille de l'image. Ainsi, le même
        label fonctionne quelle que soit la résolution.

        ### 🧮 Comment calcule-t-on ces nombres ?

        Pour une image de **W** pixels de large et **H** pixels de haut, avec un
        objet dont la boîte va de (x₁, y₁) à (x₂, y₂) :

        $$x_{center} = \\frac{(x_1 + x_2) / 2}{W} \\qquad y_{center} = \\frac{(y_1 + y_2) / 2}{H}$$

        $$width = \\frac{x_2 - x_1}{W} \\qquad height = \\frac{y_2 - y_1}{H}$$
        """
    )

    st.markdown("### 🎯 Calculons un label en direct")
    st.caption(
        "Déplacez les curseurs pour placer un objet sur l'image. "
        "Le label YOLO correspondant est recalculé à chaque mouvement."
    )

    img_size = 400
    col_a, col_b = st.columns([1, 1])
    with col_a:
        x_center_px = st.slider("Centre X (pixels)", 30, img_size - 30, 200)
        y_center_px = st.slider("Centre Y (pixels)", 30, img_size - 30, 200)
        box_w = st.slider("Largeur (pixels)", 20, 300, 120)
        box_h = st.slider("Hauteur (pixels)", 20, 300, 80)
        class_id = st.selectbox(
            "Classe", [0, 1, 2],
            format_func=lambda c: f"{c} — {CLASS_NAMES[c]}",
        )

    x1, y1 = x_center_px - box_w // 2, y_center_px - box_h // 2
    x2, y2 = x_center_px + box_w // 2, y_center_px + box_h // 2
    xc_norm = ((x1 + x2) / 2) / img_size
    yc_norm = ((y1 + y2) / 2) / img_size
    w_norm = (x2 - x1) / img_size
    h_norm = (y2 - y1) / img_size

    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
    color_bgr = {
        0: (107, 107, 255),   # rouge
        1: (196, 205, 78),    # turquoise
        2: (102, 209, 255),   # jaune
    }[class_id]
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color_bgr, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (50, 50, 50), 2)
    cv2.circle(canvas, (x_center_px, y_center_px), 5, (0, 0, 0), -1)
    cv2.putText(canvas, "centre", (x_center_px + 8, y_center_px - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    with col_b:
        st.image(canvas, caption=f"Image {img_size}×{img_size} pixels")

    st.markdown("#### 📋 Label YOLO calculé")
    label_str = f"{class_id} {xc_norm:.6f} {yc_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
    st.code(label_str, language="text")

    st.markdown("#### 🔍 Détail du calcul")
    detail = pd.DataFrame({
        "Valeur": ["x_center", "y_center", "width", "height"],
        "Pixels": [f"{(x1+x2)/2:.0f}", f"{(y1+y2)/2:.0f}",
                   f"{x2-x1}", f"{y2-y1}"],
        "Division par": [img_size] * 4,
        "Résultat normalisé (0-1)": [
            f"{xc_norm:.4f}", f"{yc_norm:.4f}",
            f"{w_norm:.4f}", f"{h_norm:.4f}",
        ],
    })
    st.dataframe(detail, hide_index=True, use_container_width=True)

    st.warning(
        "⚠️ **Pièges courants à éviter :**\n"
        "- ne pas confondre `(x1, y1, x2, y2)` avec `(x_center, y_center, w, h)`\n"
        "- oublier de **normaliser** par la taille de l'image\n"
        "- décaler les `class_id` (YOLO commence à **0**, pas à 1)\n"
        "- inverser X et Y (les images ont l'origine en **haut à gauche**)"
    )


# ===========================================================================
# SECTION 6 : TRAIN / VALIDATION / TEST
# ===========================================================================
elif section == SECTIONS[6]:
    st.title("6. Train / Validation / Test")

    st.markdown(
        """
        On ne donne **jamais** toutes ses données au modèle pour l'entraînement !
        On les sépare en plusieurs ensembles ayant chacun un rôle bien précis :

        | Ensemble | Rôle | Proportion typique |
        |----------|------|--------------------|
        | **Train** (entraînement) | Le modèle **apprend** dessus | ≈ 70–80 % |
        | **Validation** | On **règle les hyperparamètres** et on suit la progression | ≈ 10–20 % |
        | **Test** | On **mesure les performances finales** sur des données encore jamais vues | ≈ 10–20 % |

        Dans ce tutoriel, on n'utilise que **train + val** (pas de test) pour
        simplifier, mais en projet réel un set de test est indispensable.
        """
    )

    st.markdown("### 📊 Visualisation des splits")
    total = st.slider("Nombre total d'images", 50, 500, 100, step=10)
    pct_train = st.slider("% pour le train", 50, 90, 70)
    pct_val = st.slider("% pour la validation", 5, 30, 15)
    pct_test = 100 - pct_train - pct_val

    if pct_test < 0:
        st.error("La somme train + val ne doit pas dépasser 100 % !")
    else:
        fig, ax = plt.subplots(figsize=(10, 1.5))
        n_train_split = int(total * pct_train / 100)
        n_val_split = int(total * pct_val / 100)
        n_test_split = total - n_train_split - n_val_split
        ax.barh(0, n_train_split, color="#4ECDC4", label=f"Train ({n_train_split})")
        ax.barh(0, n_val_split, left=n_train_split, color="#FFD166",
                label=f"Val ({n_val_split})")
        ax.barh(0, n_test_split, left=n_train_split + n_val_split,
                color="#FF6B6B", label=f"Test ({n_test_split})")
        ax.set_xlim(0, total)
        ax.set_yticks([])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=3)
        ax.set_xlabel("Nombre d'images")
        st.pyplot(fig)

    st.markdown("### ⚠️ Le risque du sur-apprentissage (overfitting)")
    st.markdown(
        """
        Si on **n'avait pas** de set de validation, on ne pourrait pas savoir si
        le modèle **généralise** ou s'il a juste **mémorisé** les exemples
        d'entraînement.

        Voici à quoi ça ressemble en pratique :
        """
    )

    epochs = np.arange(1, 31)
    train_loss = 1.0 * np.exp(-epochs / 8) + 0.05
    good_val_loss = 1.0 * np.exp(-epochs / 10) + 0.15
    overfit_val_loss = np.where(
        epochs <= 12,
        1.0 * np.exp(-epochs / 8) + 0.15,
        0.18 + 0.015 * (epochs - 12),
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, train_loss, label="Train", color="#4ECDC4", linewidth=2)
    ax1.plot(epochs, good_val_loss, label="Validation", color="#FFD166", linewidth=2)
    ax1.set_title("✅ Bon apprentissage")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_loss, label="Train", color="#4ECDC4", linewidth=2)
    ax2.plot(epochs, overfit_val_loss, label="Validation", color="#FF6B6B", linewidth=2)
    ax2.axvline(x=12, color="gray", linestyle="--", alpha=0.5)
    ax2.text(12.5, 0.6, "← le modèle commence\nà sur-apprendre", fontsize=9)
    ax2.set_title("⚠️ Sur-apprentissage (overfitting)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(alpha=0.3)
    st.pyplot(fig)

    st.markdown(
        """
        - **À gauche** : train et val baissent ensemble → le modèle apprend des
          patterns **généralisables**. ✅
        - **À droite** : la loss train continue de baisser mais la loss val
          **remonte**. Le modèle mémorise les images d'entraînement sans plus
          rien apprendre d'utile. ⚠️

        ### 🛡️ Les parades

        - **Early stopping** : on arrête l'entraînement quand la loss de validation
          stagne (option `patience` dans YOLO)
        - **Plus de données** : augmenter et diversifier le dataset
        - **Data augmentation** : appliquer des transformations aléatoires aux images
          (rotations, flips, changements de couleurs)
        - **Régularisation** : pénaliser la complexité du modèle
        """
    )


# ===========================================================================
# SECTION 7 : EPOCH, BATCH ET LOSS
# ===========================================================================
elif section == SECTIONS[7]:
    st.title("7. Epoch, batch et loss")

    st.markdown(
        """
        Trois mots qui reviennent en permanence quand on entraîne un réseau de
        neurones. Démystifions-les.
        """
    )

    st.markdown("### 📦 Batch")
    st.markdown(
        """
        Un **batch** (ou « lot ») est un **petit paquet d'images** que le modèle
        traite ensemble avant de mettre à jour ses poids.

        - Trop petit (batch=1) : mise à jour bruitée, apprentissage instable
        - Trop grand (batch=128) : nécessite beaucoup de mémoire
        - Typique sur CPU : 4 à 16
        """
    )

    n_images = 40
    col1, col2 = st.columns([1, 2])
    with col1:
        batch_size = st.slider("Taille de batch", 1, 16, 4)
    n_batches = (n_images + batch_size - 1) // batch_size
    with col2:
        st.metric("Nombre de batches par epoch", n_batches,
                  help=f"{n_images} images ÷ batch de {batch_size}")

    fig, ax = plt.subplots(figsize=(12, 2))
    for i in range(n_images):
        batch_idx = i // batch_size
        color = plt.cm.Set3(batch_idx % 12)
        ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=color))
        ax.text(i + 0.45, 0.5, str(i), ha="center", va="center", fontsize=7)
    ax.set_xlim(0, n_images)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title(f"{n_images} images regroupées en {n_batches} batches "
                 f"(chaque couleur = un batch)")
    st.pyplot(fig)

    st.markdown("### 🔁 Epoch")
    st.markdown(
        """
        Une **epoch** correspond à **un passage complet** du modèle sur **toutes
        les images** du jeu d'entraînement.

        À chaque epoch, le modèle voit donc chaque image **une fois**. On enchaîne
        plusieurs epochs (10, 30, 100...) jusqu'à ce que les performances arrêtent
        de s'améliorer.

        ```
        for epoch in range(N_EPOCHS):
            for batch in train_dataloader:    ← un epoch = parcourir TOUS les batches
                predictions = model(batch)
                loss = compute_loss(predictions, labels)
                update_weights(loss)
        ```
        """
    )

    n_epochs_demo = st.slider("Combien d'epochs ?", 1, 50, 10)
    total_updates = n_epochs_demo * n_batches
    st.info(
        f"Pour **{n_epochs_demo} epochs** avec un batch de **{batch_size}** sur "
        f"**{n_images} images**, le modèle effectue "
        f"**{total_updates} mises à jour de poids** au total."
    )

    st.markdown("### 📉 Loss (fonction de coût)")
    st.markdown(
        """
        La **loss** est un **nombre qui mesure à quel point le modèle se trompe**.
        Plus elle est petite, mieux c'est. L'entraînement consiste à modifier
        les poids du modèle pour **faire descendre la loss**.

        Pour la détection, YOLO combine **trois losses** :
        - **box loss** : erreur sur les **coordonnées** des boîtes
        - **class loss** : erreur sur la **classification** (rectangle vs cercle vs...)
        - **DFL loss** (*Distribution Focal Loss*) : affinement de la localisation

        Loss totale = box_loss + class_loss + dfl_loss
        """
    )

    epochs_arr = np.arange(1, 31)
    box_loss = 1.5 * np.exp(-epochs_arr / 6) + 0.3
    cls_loss = 2.0 * np.exp(-epochs_arr / 8) + 0.2
    dfl_loss = 1.2 * np.exp(-epochs_arr / 7) + 0.25
    total = box_loss + cls_loss + dfl_loss

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs_arr, box_loss, label="Box loss", linewidth=2)
    ax.plot(epochs_arr, cls_loss, label="Class loss", linewidth=2)
    ax.plot(epochs_arr, dfl_loss, label="DFL loss", linewidth=2)
    ax.plot(epochs_arr, total, label="Loss totale", linewidth=3, linestyle="--",
            color="black")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Exemple de courbes de loss durant l'entraînement")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.success(
        "**À retenir :**\n"
        "- **Batch** = petit paquet d'images traitées ensemble\n"
        "- **Epoch** = un passage complet sur toutes les données d'entraînement\n"
        "- **Loss** = score d'erreur qu'on cherche à minimiser"
    )


# ===========================================================================
# SECTION 8 : ENTRAÎNEMENT DU MODÈLE
# ===========================================================================
elif section == SECTIONS[8]:
    st.title("8. Entraînement du modèle")

    st.markdown(
        f"""
        On va maintenant lancer un **vrai entraînement** sur la machine.
        On part du modèle **`{PRETRAINED_MODEL}`** — la plus petite variante YOLO
        disponible chez Ultralytics (~2,6 M paramètres) — **pré-entraîné sur COCO**
        (80 classes d'objets courants), et on le **fine-tune** sur notre dataset.

        ### 🔄 L'apprentissage par transfert

        Le modèle pré-entraîné a déjà appris à reconnaître des contours, des textures,
        des formes simples grâce à des millions d'images. On garde tout ça et on
        ajuste seulement la **tête de détection** pour nos 3 classes
        (rectangle, cercle, triangle).
        """
    )

    dataset_ready = (DATASET_PATH / "data.yaml").exists()
    if not dataset_ready:
        st.warning(
            "⚠️ Aucun dataset trouvé. Allez d'abord à la **section 4** pour générer le jeu de données."
        )

    st.markdown("### 🎛️ Paramètres d'entraînement")

    col1, col2, col3 = st.columns(3)
    with col1:
        epochs = st.slider("Nombre d'epochs", 1, 50, 15,
                           help="Plus = meilleur apprentissage mais plus long")
    with col2:
        batch = st.slider("Taille de batch", 1, 16, 4)
    with col3:
        imgsz = st.select_slider("Taille d'image",
                                 options=[96, 128, 160, 224, 320, 416, 640],
                                 value=160,
                                 help="Plus grand = plus précis mais plus lent. "
                                      "Le levier principal pour gagner du temps !")

    col4, col5 = st.columns(2)
    with col4:
        patience = st.slider("Patience (early stopping)", 0, 20, 10,
                             help="Arrêt anticipé si pas d'amélioration")
    with col5:
        workers = st.slider("Workers (threads de chargement)", 0, 4, 2)

    # Estimation grossière : ~ (imgsz / 160)^2 * 6 s par epoch sur CPU moderne
    sec_per_epoch = max(2, int((imgsz / 160) ** 2 * 6))
    estimated_time = epochs * sec_per_epoch
    st.info(
        f"⏱️ **Estimation de durée** : ~{estimated_time // 60} min "
        f"{estimated_time % 60}s sur un CPU moderne "
        f"(modèle **YOLO11n**, imgsz={imgsz}, {epochs} epochs). "
        "L'app va **bloquer** pendant l'entraînement, soyez patients !"
    )
    st.caption(
        "💡 **Pour aller plus vite** : baissez `imgsz` (impact ×4 si vous passez "
        "de 320 à 160), réduisez `epochs`, ou les deux. Le choix du modèle "
        "(YOLO11n vs YOLOv8n) joue beaucoup moins."
    )

    if st.button("🚀 Lancer l'entraînement", type="primary", disabled=not dataset_ready):
        st.markdown("### 📊 Entraînement en cours")
        log_placeholder = st.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            from ultralytics import YOLO

            status_text.text(f"⏳ Chargement du modèle {PRETRAINED_MODEL} pré-entraîné...")
            model = YOLO(PRETRAINED_MODEL)
            status_text.text("✅ Modèle chargé. Démarrage de l'entraînement...")

            captured_output = io.StringIO()

            import contextlib

            with contextlib.redirect_stdout(captured_output):
                results = model.train(
                    data=str(DATASET_PATH / "data.yaml"),
                    epochs=epochs,
                    imgsz=imgsz,
                    batch=batch,
                    device="cpu",
                    workers=workers,
                    patience=patience,
                    project=str(RUNS_PATH),
                    name=RUN_NAME,
                    exist_ok=True,
                    verbose=True,
                    plots=True,
                )

            progress_bar.progress(100)
            status_text.text("✅ Entraînement terminé !")

            log_placeholder.code(captured_output.getvalue()[-3000:], language="text")

            st.success(
                f"🎉 Entraînement terminé ! Résultats dans `{results.save_dir}`"
            )

            results_png = Path(results.save_dir) / "results.png"
            if results_png.exists():
                st.markdown("### 📈 Courbes d'entraînement")
                st.image(str(results_png),
                         caption="Évolution des losses et des métriques")
            st.balloons()
        except Exception as e:
            st.error(f"❌ Erreur durant l'entraînement : {e}")

    run_dir = find_latest_run()
    if run_dir is not None:
        st.markdown("---")
        st.markdown("### 📂 Dernier entraînement disponible")
        st.write(f"**Dossier** : `{run_dir}`")
        results_png = run_dir / "results.png"
        if results_png.exists():
            st.image(str(results_png),
                     caption="Courbes de l'entraînement précédent")


# ===========================================================================
# SECTION 9 : MÉTRIQUES D'ÉVALUATION
# ===========================================================================
elif section == SECTIONS[9]:
    st.title("9. Métriques d'évaluation")

    st.markdown(
        """
        Comment savoir si un détecteur d'objets est **bon** ? On utilise plusieurs
        métriques complémentaires.
        """
    )

    st.markdown("### 🧮 IoU — Intersection over Union")
    st.markdown(
        """
        L'**IoU** mesure le **recouvrement** entre une boîte prédite et une boîte
        de vérité terrain :

        $$IoU = \\frac{\\text{Aire d'intersection}}{\\text{Aire d'union}}$$

        - IoU = 0 : aucun recouvrement
        - IoU = 1 : recouvrement parfait
        - On considère souvent **IoU > 0,5** comme une « bonne » détection
        """
    )

    st.caption("Déplacez les curseurs pour faire varier les deux boîtes et observer l'IoU.")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("**Boîte verte (vérité terrain)**")
        gt_x = st.slider("Position X (verte)", 0, 300, 80)
        gt_y = st.slider("Position Y (verte)", 0, 300, 80)
        gt_w = st.slider("Largeur (verte)", 40, 250, 140)
        gt_h = st.slider("Hauteur (verte)", 40, 250, 120)

        st.markdown("**Boîte rouge (prédiction)**")
        pr_x = st.slider("Position X (rouge)", 0, 300, 140)
        pr_y = st.slider("Position Y (rouge)", 0, 300, 120)
        pr_w = st.slider("Largeur (rouge)", 40, 250, 130)
        pr_h = st.slider("Hauteur (rouge)", 40, 250, 110)

    canvas_iou = np.ones((400, 400, 3), dtype=np.uint8) * 240
    cv2.rectangle(canvas_iou, (gt_x, gt_y), (gt_x + gt_w, gt_y + gt_h),
                  (100, 200, 100), 3)
    cv2.rectangle(canvas_iou, (pr_x, pr_y), (pr_x + pr_w, pr_y + pr_h),
                  (100, 100, 255), 3)

    iou_val = compute_iou(
        (gt_x, gt_y, gt_x + gt_w, gt_y + gt_h),
        (pr_x, pr_y, pr_x + pr_w, pr_y + pr_h),
    )

    with col2:
        st.image(canvas_iou, caption="🟢 vérité terrain  🔴 prédiction")
        st.metric("IoU", f"{iou_val:.3f}")
        if iou_val > 0.7:
            st.success("✅ Excellente détection")
        elif iou_val > 0.5:
            st.info("👍 Détection acceptable")
        elif iou_val > 0.0:
            st.warning("⚠️ Détection insuffisante")
        else:
            st.error("❌ Aucun recouvrement")

    st.markdown("---")
    st.markdown("### 📊 Précision et Rappel")
    st.markdown(
        """
        Une fois qu'on a décidé qu'une boîte est « correcte » (IoU > seuil), on
        peut compter :

        - **Vrais positifs (VP / TP)** : objets correctement détectés
        - **Faux positifs (FP)** : détections qui ne correspondent à aucun objet
        - **Faux négatifs (FN)** : objets ratés (pas détectés)

        On définit alors :

        $$\\text{Précision} = \\frac{VP}{VP + FP} \\quad \\text{(parmi mes détections, combien sont correctes ?)}$$

        $$\\text{Rappel} = \\frac{VP}{VP + FN} \\quad \\text{(parmi les vrais objets, combien ai-je trouvés ?)}$$
        """
    )

    col_p, col_r = st.columns(2)
    with col_p:
        vp = st.number_input("Vrais positifs (VP)", 0, 100, 8)
        fp = st.number_input("Faux positifs (FP)", 0, 100, 2)
        fn = st.number_input("Faux négatifs (FN)", 0, 100, 1)
        precision = vp / (vp + fp) if (vp + fp) > 0 else 0
        recall = vp / (vp + fn) if (vp + fn) > 0 else 0
    with col_r:
        st.metric("Précision", f"{precision:.3f}")
        st.metric("Rappel", f"{recall:.3f}")
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        st.metric("F1-score", f"{f1:.3f}",
                  help="Moyenne harmonique de précision et rappel")

    st.markdown(
        """
        ### ⚖️ Le compromis précision / rappel

        - Si on **abaisse le seuil de confiance** → plus de détections → souvent
          **plus de rappel** mais **moins de précision** (on prend tout, on se
          trompe plus souvent).
        - Si on **monte le seuil** → moins de détections → **plus de précision**
          mais **moins de rappel** (on ne garde que les détections très sûres,
          on en rate plus).
        """
    )

    st.markdown("---")
    st.markdown("### 🏆 mAP — Mean Average Precision")
    st.markdown(
        """
        La **mAP** est **la métrique reine** de la détection d'objets. Elle
        combine précision et rappel sur **tous les seuils de confiance** possibles,
        moyennés sur **toutes les classes**.

        - **mAP@50** (ou mAP50) : moyenne avec un seuil IoU de **0,5**
        - **mAP@50-95** : moyenne sur 10 seuils IoU de 0,5 à 0,95 (par pas de 0,05).
          C'est la métrique principale de COCO, plus exigeante.

        | mAP50 | Qualité |
        |-------|---------|
        | > 0,9 | 🥇 excellent |
        | 0,7 – 0,9 | 🥈 très bon |
        | 0,5 – 0,7 | 🥉 correct |
        | < 0,5 | 🔧 à améliorer |
        """
    )


# ===========================================================================
# SECTION 10 : INFÉRENCE
# ===========================================================================
elif section == SECTIONS[10]:
    st.title("10. Inférence sur de nouvelles images")

    st.markdown(
        """
        L'**inférence**, c'est l'étape où on **utilise** le modèle entraîné pour
        détecter des objets sur de **nouvelles images**.

        On peut régler :
        - **conf** : le seuil de confiance (entre 0 et 1). Plus bas → plus de
          détections (et plus de faux positifs)
        - **iou** : le seuil pour la NMS (Non-Maximum Suppression), qui
          fusionne les boîtes qui se chevauchent trop
        """
    )

    run_dir = find_latest_run()
    if run_dir is None:
        st.warning(
            "⚠️ Aucun modèle entraîné trouvé. "
            "Allez d'abord à la **section 8** pour entraîner un modèle."
        )
    else:
        weights = run_dir / "weights" / "best.pt"
        st.success(f"✅ Modèle disponible : `{weights}`")

        col1, col2 = st.columns(2)
        with col1:
            conf_thresh = st.slider("Seuil de confiance (conf)", 0.05, 0.95, 0.25, 0.05)
        with col2:
            iou_thresh = st.slider("Seuil IoU pour NMS", 0.1, 0.9, 0.45, 0.05)

        st.markdown("### 🖼️ Choisir une image à analyser")
        source_choice = st.radio(
            "Source de l'image",
            ["Image de validation existante", "Image synthétique aléatoire",
             "Téléversement (upload)"],
            horizontal=True,
        )

        img_array = None
        img_label = ""

        if source_choice == "Image de validation existante":
            val_dir = DATASET_PATH / "images" / "val"
            if val_dir.exists():
                val_images = sorted(val_dir.glob("*.jpg"))
                if val_images:
                    picked = st.selectbox("Image", val_images,
                                          format_func=lambda p: p.name)
                    img_array = np.array(Image.open(picked))
                    img_label = picked.name
                else:
                    st.warning("Pas d'images dans le dossier de validation.")
            else:
                st.warning("Dossier de validation introuvable.")

        elif source_choice == "Image synthétique aléatoire":
            seed_inf = st.slider("Graine aléatoire", 0, 10_000, 2025)
            n_obj = st.slider("Nombre d'objets", 1, 6, (1, 3))
            img_array, _ = make_synthetic_image(seed_inf, tuple(n_obj))
            img_label = f"synthetic_{seed_inf}.jpg"

        else:
            uploaded = st.file_uploader("Téléversez une image (jpg, png)",
                                        type=["jpg", "jpeg", "png"])
            if uploaded is not None:
                img_array = np.array(Image.open(uploaded).convert("RGB"))
                img_label = uploaded.name

        if img_array is not None and st.button("🔮 Détecter les objets", type="primary"):
            with st.spinner("Inférence en cours..."):
                from ultralytics import YOLO
                model = YOLO(str(weights))
                results = model.predict(
                    source=img_array,
                    conf=conf_thresh,
                    iou=iou_thresh,
                    device="cpu",
                    verbose=False,
                )
            result = results[0]
            annotated = result.plot()[:, :, ::-1]

            col_a, col_b = st.columns(2)
            with col_a:
                st.image(img_array, caption=f"Image originale : {img_label}")
            with col_b:
                st.image(annotated, caption=f"Détections (conf ≥ {conf_thresh})")

            st.markdown(f"### 🎯 {len(result.boxes)} objet(s) détecté(s)")
            if len(result.boxes) > 0:
                rows = []
                for i, box in enumerate(result.boxes):
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                    rows.append({
                        "#": i + 1,
                        "Classe": CLASS_NAMES.get(cls, f"id={cls}"),
                        "Confiance": f"{conf:.3f}",
                        "x1": f"{x1:.0f}", "y1": f"{y1:.0f}",
                        "x2": f"{x2:.0f}", "y2": f"{y2:.0f}",
                    })
                st.dataframe(pd.DataFrame(rows), hide_index=True,
                             use_container_width=True)


# ===========================================================================
# SECTION 11 : RÉCAPITULATIF
# ===========================================================================
else:
    st.title("11. Récapitulatif")

    st.markdown(
        """
        Bravo, vous avez parcouru toutes les étapes du **fine-tuning** d'un
        détecteur d'objets YOLO ! 🎉
        """
    )

    st.markdown("### ✅ Ce que vous avez vu")
    st.markdown(
        """
        | Étape | Concept clé |
        |-------|-------------|
        | Détection d'objets | Quoi **et** où, plusieurs objets par image |
        | Principe YOLO | Une seule passe, grille de prédiction, très rapide |
        | Dataset | Images + labels au format YOLO |
        | Labels | `class_id xc yc w h` normalisés entre 0 et 1 |
        | Train / Val / Test | Mesurer la généralisation, éviter le sur-apprentissage |
        | Epoch / Batch / Loss | La mécanique de l'apprentissage |
        | Fine-tuning | Partir d'un modèle pré-entraîné, l'adapter à nos classes |
        | Métriques | IoU, précision, rappel, mAP |
        | Inférence | Utiliser le modèle entraîné sur de nouvelles images |
        """
    )

    st.markdown("### 🚀 Pour aller plus loin")
    st.markdown(
        """
        - **Collecter votre propre dataset** avec votre téléphone et l'outil [LabelImg](https://github.com/tzutalin/labelImg)
        - **Essayer un modèle plus gros** : YOLOv8s, YOLOv8m... (besoin de plus de RAM/GPU)
        - **Augmenter les données** : rotations, flips, modifications de couleur
        - **Détection vidéo** : appliquer le modèle frame par frame sur une vidéo
        - **Déploiement** : exporter en ONNX, TFLite ou CoreML pour mobile
        - **Autres tâches** : segmentation d'instances, pose estimation, classification (toutes disponibles dans Ultralytics)
        """
    )

    st.markdown("### 📚 Ressources")
    st.markdown(
        """
        - 📖 [Documentation officielle Ultralytics](https://docs.ultralytics.com/)
        - 📄 [Article original YOLO](https://arxiv.org/abs/1506.02640)
        - 🛠️ [Roboflow](https://roboflow.com/) — annotation et gestion de datasets
        - 💬 [Forum communautaire](https://github.com/ultralytics/ultralytics/discussions)
        """
    )

    st.success(
        "🎓 Vous avez maintenant tous les outils pour démarrer vos propres "
        "projets de détection d'objets. Bonne route !"
    )
