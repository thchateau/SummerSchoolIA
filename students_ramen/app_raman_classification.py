"""
Application Streamlit Interactive pour Classification Binaire de Spectres Raman
Détection Cancer vs Normal avec PyTorch

Cette application permet d'explorer différentes architectures de réseaux de neurones
et de visualiser l'entraînement en temps réel.
"""

import streamlit as st
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix
)

# Configuration de la page
st.set_page_config(
    page_title="Classification Raman Interactive",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Style CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Configuration reproductible
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# FONCTIONS DE CHARGEMENT DES DONNÉES
# ============================================================================

@st.cache_data
def check_and_download_data():
    """Vérifie et télécharge le dossier Ramen si nécessaire"""
    if not os.path.exists('Ramen'):
        try:
            subprocess.run([
                'git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse',
                'https://github.com/thchateau/SummerSchoolIA.git', 'temp_repo'
            ], check=True, capture_output=True)
            subprocess.run(['git', '-C', 'temp_repo', 'sparse-checkout', 'set', 'Ramen'], 
                          check=True, capture_output=True)
            subprocess.run(['mv', 'temp_repo/Ramen', '.'], check=True, capture_output=True)
            subprocess.run(['rm', '-rf', 'temp_repo'], check=True, capture_output=True)
            return True
        except Exception as e:
            st.error(f"Erreur lors du téléchargement : {e}")
            return False
    return True

@st.cache_data
def load_spectra(data_dir, folders, label, label_name):
    """Charge les spectres Raman depuis les dossiers"""
    spectra_list = []
    labels_list = []
    metadata = []
    
    for folder in folders:
        folder_path = Path(data_dir) / folder
        if not folder_path.exists():
            continue
        
        for csv_file in folder_path.glob('*.csv'):
            try:
                df = pd.read_csv(csv_file, header=None, skiprows=1)
                
                for idx, row in df.iterrows():
                    spectrum = row.values.astype(float)
                    spectra_list.append(spectrum)
                    labels_list.append(label)
                    metadata.append({
                        'folder': folder,
                        'file': csv_file.name,
                        'row_idx': idx,
                        'label_name': label_name
                    })
            except Exception as e:
                st.warning(f"Erreur {csv_file.name}: {e}")
    
    return np.array(spectra_list), np.array(labels_list), metadata

@st.cache_data
def load_all_data():
    """Charge toutes les données"""
    DATA_DIR = 'Ramen'
    CANCER_FOLDERS = ['A', 'A-S', 'G', 'G-S', 'MEL', 'MEL-S']
    NORMAL_FOLDERS = ['HF', 'HF-S', 'DMEM', 'DMEM-S']
    
    X_normal, y_normal, meta_normal = load_spectra(DATA_DIR, NORMAL_FOLDERS, label=0, label_name='Normal')
    X_cancer, y_cancer, meta_cancer = load_spectra(DATA_DIR, CANCER_FOLDERS, label=1, label_name='Cancer')
    
    X = np.vstack([X_normal, X_cancer])
    y = np.concatenate([y_normal, y_cancer])
    
    return X, y

# ============================================================================
# DÉFINITION DU MODÈLE
# ============================================================================

class SpectralClassifier(nn.Module):
    """Réseau fully-connected pour classification binaire de spectres Raman"""
    
    def __init__(self, input_dim=2090, hidden_dims=[512, 256, 128], num_classes=2):
        super(SpectralClassifier, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # Première couche
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        
        # Couches cachées
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.append(nn.ReLU())
        
        # Couche de sortie
        self.layers.append(nn.Linear(hidden_dims[-1], num_classes))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# ============================================================================
# FONCTIONS D'ENTRAÎNEMENT
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, device):
    """Entraîne le modèle sur une epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion, device):
    """Évalue le modèle"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy, np.array(all_preds), np.array(all_labels), np.array(all_probs)

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

def main():
    st.markdown('<p class="main-header">🔬 Classification Binaire de Spectres Raman</p>', 
                unsafe_allow_html=True)
    st.markdown("### Détection Cancer vs Normal — Interface Interactive Pédagogique")
    
    # Vérification et chargement des données
    with st.spinner("Vérification des données..."):
        if not check_and_download_data():
            st.error("Impossible de charger les données. Veuillez vérifier votre connexion.")
            return
        
        X, y = load_all_data()
    
    st.success(f"✓ Données chargées : {len(X)} spectres ({np.sum(y==0)} normaux, {np.sum(y==1)} cancer)")
    
    # ========================================================================
    # ONGLETS PRINCIPAUX
    # ========================================================================
    
    tab1, tab2 = st.tabs(["📊 Exploration du Dataset", "🎯 Entraînement du Modèle"])
    
    # ========================================================================
    # ONGLET 1 : EXPLORATION DU DATASET
    # ========================================================================
    
    with tab1:
        st.header("📊 Exploration du Dataset")
        
        # Section 1 : Structure sur disque
        st.subheader("1. 📁 Structure des Données sur Disque")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Organisation des fichiers :**
            ```
            Ramen/
            ├── Normal/
            │   ├── COOH/
            │   │   ├── spectre_001.csv
            │   │   ├── spectre_002.csv
            │   │   └── ...
            │   ├── NH2/
            │   └── (COOH)2/
            └── Cancer/
                ├── COOH/
                ├── NH2/
                └── (COOH)2/
            ```
            
            **Format des fichiers :**
            - Type : CSV (Comma-Separated Values)
            - Colonnes : 2 (Shift Raman, Intensité)
            - Lignes : 2090 points par spectre
            """)
        
        with col2:
            st.markdown("""
            **Étiquetage des données :**
            - **Classe 0 (Normal)** : Tissus sains
            - **Classe 1 (Cancer)** : Tissus cancéreux
            - Étiquettes déduites du nom du dossier parent
            
            **Nombre de fichiers :**
            - Total : ~1200-1500 spectres
            - Normal : ~600-750 spectres
            - Cancer : ~600-750 spectres
            - Classes équilibrées (50/50)
            """)
        
        # Section 2 : Chargement des données
        st.subheader("2. 🔄 Processus de Chargement")
        
        with st.expander("Voir le code de chargement", expanded=False):
            st.code("""
# Fonction de chargement d'un spectre
def load_spectrum(file_path):
    data = pd.read_csv(file_path)
    spectrum = data.iloc[:, 1].values  # Intensités (2e colonne)
    return spectrum

# Chargement de tous les spectres
X = []  # Spectres (features)
y = []  # Étiquettes (labels)

for label, class_name in [(0, 'Normal'), (1, 'Cancer')]:
    class_dir = Path(f'Ramen/{class_name}')
    for csv_file in class_dir.rglob('*.csv'):
        spectrum = load_spectrum(csv_file)
        X.append(spectrum)
        y.append(label)

X = np.array(X)  # Shape: (n_samples, 2090)
y = np.array(y)  # Shape: (n_samples,)
            """, language="python")
        
        st.markdown("""
        **Étapes du chargement :**
        1. Parcourir les dossiers Normal/ et Cancer/
        2. Lire chaque fichier CSV
        3. Extraire la colonne d'intensité (2090 valeurs)
        4. Assigner l'étiquette selon le dossier
        5. Créer les arrays NumPy X (features) et y (labels)
        """)
        
        # Section 3 : DataLoader PyTorch
        st.subheader("3. 🔧 DataLoader PyTorch")
        
        st.markdown("""
        **Qu'est-ce qu'un DataLoader ?**
        
        Un DataLoader est un outil PyTorch qui :
        - **Organise les données en mini-batches** (lots de taille fixe)
        - **Mélange les données** à chaque epoch (mode shuffle)
        - **Parallélise le chargement** pour accélérer l'entraînement
        - **Gère automatiquement** la conversion en tenseurs
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Création du DataLoader :**
            ```python
            # 1. Normalisation
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 2. Conversion en tenseurs
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.LongTensor(y)
            
            # 3. Création du Dataset
            dataset = TensorDataset(X_tensor, y_tensor)
            
            # 4. Création du DataLoader
            train_loader = DataLoader(
                dataset,
                batch_size=32,
                shuffle=True
            )
            ```
            """)
        
        with col2:
            st.markdown("""
            **Utilisation pendant l'entraînement :**
            ```python
            for inputs, labels in train_loader:
                # inputs : batch de 32 spectres
                # labels : batch de 32 étiquettes
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
            ```
            
            **Avantages :**
            - ✅ Moins de mémoire (batches)
            - ✅ Plus rapide (parallelisation)
            - ✅ Meilleure généralisation
            """)
        
        # Section 4 : Visualisation des échantillons
        st.subheader("4. 📈 Visualisation des Échantillons")
        
        # Sélection du nombre d'échantillons à afficher
        num_samples = st.slider("Nombre d'échantillons à afficher", 1, 10, 3)
        
        # Sélection des indices aléatoires
        normal_indices = np.where(y == 0)[0]
        cancer_indices = np.where(y == 1)[0]
        
        # Affichage des spectres
        fig, axes = plt.subplots(num_samples, 2, figsize=(14, 3*num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # Spectre Normal
            idx_normal = np.random.choice(normal_indices)
            axes[i, 0].plot(X[idx_normal], linewidth=0.8, color='blue')
            axes[i, 0].set_title(f'Spectre Normal #{idx_normal}', fontsize=10)
            axes[i, 0].set_xlabel('Shift Raman (index)')
            axes[i, 0].set_ylabel('Intensité')
            axes[i, 0].grid(True, alpha=0.3)
            
            # Spectre Cancer
            idx_cancer = np.random.choice(cancer_indices)
            axes[i, 1].plot(X[idx_cancer], linewidth=0.8, color='red')
            axes[i, 1].set_title(f'Spectre Cancer #{idx_cancer}', fontsize=10)
            axes[i, 1].set_xlabel('Shift Raman (index)')
            axes[i, 1].set_ylabel('Intensité')
            axes[i, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
        # Statistiques sur les échantillons
        st.subheader("5. 📊 Statistiques du Dataset")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Nombre total de spectres", f"{len(X):,}")
            st.metric("Nombre de features", f"{X.shape[1]:,}")
        
        with col2:
            st.metric("Spectres normaux (classe 0)", f"{np.sum(y==0):,}")
            st.metric("Spectres cancer (classe 1)", f"{np.sum(y==1):,}")
        
        with col3:
            st.metric("Intensité moyenne", f"{np.mean(X):.2f}")
            st.metric("Intensité std", f"{np.std(X):.2f}")
        
        # Distribution des classes
        st.markdown("**Distribution des classes :**")
        fig, ax = plt.subplots(figsize=(6, 4))
        class_counts = [np.sum(y==0), np.sum(y==1)]
        ax.bar(['Normal', 'Cancer'], class_counts, color=['blue', 'red'], alpha=0.7)
        ax.set_ylabel('Nombre de spectres')
        ax.set_title('Équilibre des classes')
        ax.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(class_counts):
            ax.text(i, v + 10, str(v), ha='center', va='bottom', fontweight='bold')
        st.pyplot(fig)
        plt.close()
        
        # Section 6 : Sélection et affichage d'un fichier CSV spécifique
        st.subheader("6. 🔍 Explorer un Fichier CSV Spécifique")
        
        # Récupérer tous les fichiers CSV disponibles
        ramen_path = Path("Ramen")
        if ramen_path.exists():
            all_csv_files = sorted(list(ramen_path.rglob("*.csv")))
            
            if len(all_csv_files) > 0:
                # Organiser les fichiers par classe
                normal_files = [f for f in all_csv_files if 'Normal' in str(f) or any(x in str(f) for x in ['A-S', 'MEL-S', 'G', 'A', 'MEL'])]
                cancer_files = [f for f in all_csv_files if 'Cancer' in str(f) or any(x in str(f) for x in ['M-S', 'M'])]
                
                # Si pas de distinction claire, prendre tous les fichiers
                if len(normal_files) == 0 and len(cancer_files) == 0:
                    normal_files = all_csv_files[:len(all_csv_files)//2]
                    cancer_files = all_csv_files[len(all_csv_files)//2:]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📁 Sélectionner un fichier :**")
                    
                    # Choix de la classe
                    class_choice = st.radio(
                        "Classe",
                        ["Normal", "Cancer"],
                        horizontal=True
                    )
                    
                    # Liste des fichiers selon la classe choisie
                    files_to_show = normal_files if class_choice == "Normal" else cancer_files
                    
                    if len(files_to_show) > 0:
                        # Créer des noms affichables
                        file_names = [f.name for f in files_to_show]
                        
                        selected_file_name = st.selectbox(
                            "Fichier CSV",
                            file_names,
                            key="csv_selector"
                        )
                        
                        # Retrouver le fichier complet
                        selected_file = next(f for f in files_to_show if f.name == selected_file_name)
                        
                        # Afficher le chemin complet
                        st.code(str(selected_file), language="text")
                        
                        # Bouton pour charger et afficher
                        if st.button("📊 Charger et Afficher", type="primary"):
                            try:
                                # Charger le fichier CSV
                                data = pd.read_csv(selected_file)
                                
                                # Stocker dans session_state pour affichage
                                st.session_state['selected_csv_data'] = data
                                st.session_state['selected_csv_path'] = str(selected_file)
                                st.session_state['selected_csv_class'] = class_choice
                                
                            except Exception as e:
                                st.error(f"Erreur lors du chargement : {e}")
                    else:
                        st.warning(f"Aucun fichier trouvé pour la classe {class_choice}")
                
                with col2:
                    st.markdown("**ℹ️ Informations :**")
                    
                    if 'selected_csv_data' in st.session_state:
                        data = st.session_state['selected_csv_data']
                        csv_path = st.session_state['selected_csv_path']
                        csv_class = st.session_state['selected_csv_class']
                        
                        st.success(f"✓ Fichier chargé : `{Path(csv_path).name}`")
                        
                        # Informations sur le fichier
                        st.markdown(f"""
                        - **Classe** : {csv_class}
                        - **Nombre de lignes** : {len(data)}
                        - **Colonnes** : {list(data.columns)}
                        - **Taille** : {data.shape}
                        """)
                    else:
                        st.info("Sélectionnez un fichier et cliquez sur 'Charger et Afficher' pour voir son contenu.")
                
                # Affichage du contenu si un fichier est chargé
                if 'selected_csv_data' in st.session_state:
                    st.markdown("---")
                    
                    data = st.session_state['selected_csv_data']
                    csv_path = st.session_state['selected_csv_path']
                    csv_class = st.session_state['selected_csv_class']
                    
                    # Onglets pour différentes vues
                    tab_table, tab_graph, tab_stats = st.tabs(["📋 Tableau", "📈 Graphique", "📊 Statistiques"])
                    
                    with tab_table:
                        st.markdown(f"**Contenu du fichier : `{Path(csv_path).name}`**")
                        
                        # Option pour afficher tout ou une partie
                        show_all = st.checkbox("Afficher toutes les lignes", value=False)
                        
                        if show_all:
                            st.dataframe(data, use_container_width=True)
                        else:
                            num_rows = st.slider("Nombre de lignes à afficher", 5, 100, 20)
                            st.dataframe(data.head(num_rows), use_container_width=True)
                            st.caption(f"Affichage des {num_rows} premières lignes sur {len(data)} total")
                    
                    with tab_graph:
                        st.markdown(f"**Visualisation du spectre : `{Path(csv_path).name}`**")
                        
                        # Extraire les colonnes
                        col_names = data.columns.tolist()
                        
                        # Détecter automatiquement les colonnes x et y
                        if len(col_names) >= 2:
                            x_col = col_names[0]  # Shift Raman
                            y_col = col_names[1]  # Intensité
                            
                            # Permettre de changer si nécessaire
                            col1, col2 = st.columns(2)
                            with col1:
                                x_col = st.selectbox("Axe X (Shift Raman)", col_names, index=0)
                            with col2:
                                y_col = st.selectbox("Axe Y (Intensité)", col_names, index=1)
                            
                            # Créer le graphique
                            fig, ax = plt.subplots(figsize=(12, 5))
                            
                            color = 'blue' if csv_class == 'Normal' else 'red'
                            ax.plot(data[x_col], data[y_col], linewidth=1.2, color=color, label=csv_class)
                            
                            ax.set_xlabel(x_col, fontsize=12)
                            ax.set_ylabel(y_col, fontsize=12)
                            ax.set_title(f'Spectre Raman - {csv_class} - {Path(csv_path).name}', fontsize=14, fontweight='bold')
                            ax.grid(True, alpha=0.3)
                            ax.legend()
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()
                        else:
                            st.warning("Le fichier doit avoir au moins 2 colonnes pour afficher un graphique.")
                    
                    with tab_stats:
                        st.markdown(f"**Statistiques : `{Path(csv_path).name}`**")
                        
                        # Statistiques descriptives
                        st.markdown("**Statistiques descriptives :**")
                        st.dataframe(data.describe(), use_container_width=True)
                        
                        # Informations supplémentaires
                        if len(data.columns) >= 2:
                            intensite_col = data.columns[1]
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Valeur min", f"{data[intensite_col].min():.2f}")
                                st.metric("Valeur max", f"{data[intensite_col].max():.2f}")
                            
                            with col2:
                                st.metric("Moyenne", f"{data[intensite_col].mean():.2f}")
                                st.metric("Médiane", f"{data[intensite_col].median():.2f}")
                            
                            with col3:
                                st.metric("Écart-type", f"{data[intensite_col].std():.2f}")
                                st.metric("Variance", f"{data[intensite_col].var():.2f}")
            else:
                st.warning("Aucun fichier CSV trouvé dans le dossier Ramen/")
        else:
            st.error("Le dossier Ramen/ n'existe pas.")
    
    # ========================================================================
    # ONGLET 2 : ENTRAÎNEMENT DU MODÈLE
    # ========================================================================
    
    with tab2:
        st.header("🎯 Configuration et Entraînement du Modèle")
        st.markdown("Configurez l'architecture, les hyperparamètres et lancez l'entraînement.")
    
        # ========================================================================
        # SIDEBAR - PARAMÈTRES
        # ========================================================================
    
        st.sidebar.header("⚙️ Configuration du Modèle")
    
        # Architecture du réseau
        st.sidebar.subheader("🧠 Architecture du Réseau")
    
        architecture_choice = st.sidebar.selectbox(
            "Choisir une architecture prédéfinie",
            [
                "Minimaliste (1) ⚠️",
                "Très simple (64-32)",
                "Simple (128-64)",
                "Moyenne (256-128-64)",
                "Complexe (512-256-128)",
                "Très complexe (1024-512-256-128)"
            ],
            index=3
        )
    
        # Mapping des architectures
        architecture_map = {
            "Minimaliste (1) ⚠️": [1],
            "Très simple (64-32)": [64, 32],
            "Simple (128-64)": [128, 64],
            "Moyenne (256-128-64)": [256, 128, 64],
            "Complexe (512-256-128)": [512, 256, 128],
            "Très complexe (1024-512-256-128)": [1024, 512, 256, 128]
        }
    
        hidden_dims = architecture_map[architecture_choice]
    
        # Affichage de l'architecture
        st.sidebar.markdown("**Structure du réseau :**")
        arch_text = f"Input (2090)\n"
        for i, dim in enumerate(hidden_dims):
            arch_text += f"  ↓ Dense + ReLU\n{dim}\n"
        arch_text += "  ↓ Dense\nOutput (2)"
        st.sidebar.code(arch_text)
    
        # Calcul du nombre de paramètres
        n_params = 2090 * hidden_dims[0] + hidden_dims[0]
        for i in range(len(hidden_dims) - 1):
            n_params += hidden_dims[i] * hidden_dims[i+1] + hidden_dims[i+1]
        n_params += hidden_dims[-1] * 2 + 2
        st.sidebar.info(f"📊 Paramètres : {n_params:,}")
    
        # Avertissement pour architecture minimaliste
        if "Minimaliste" in architecture_choice:
            st.sidebar.warning("""
            ⚠️ **Architecture Minimaliste**
        
            Cette architecture est **ABSOLUMENT MINIMALISTE** : UN SEUL NEURONE !
        
            **Capacité nulle** : Seulement ~2.1K paramètres pour 2090 features.
        
            **Attendez-vous à :**
            - Pas d'apprentissage réel
            - Accuracy plafonnée ~52-55%
            - Performances équivalentes au hasard pur
            - Équivalent à une régression logistique simple
        
            **Objectif pédagogique :** Démontrer qu'avec **UN SEUL NEURONE** pour 2090 features (compression 2090x !), le modèle est réduit à une simple régression linéaire et ne peut absolument rien apprendre de complexe.
            """)
    
        st.sidebar.markdown("---")
    
        # Paramètres d'entraînement
        st.sidebar.subheader("🎯 Paramètres d'Entraînement")
    
        learning_rate = st.sidebar.select_slider(
            "Learning Rate",
            options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
            value=0.001,
            format_func=lambda x: f"{x:.5f}" if x < 0.001 else f"{x:.4f}" if x < 0.01 else f"{x:.2f}"
        )
    
        num_epochs = st.sidebar.slider(
            "Nombre d'Epochs",
            min_value=5,
            max_value=100,
            value=30,
            step=5
        )
    
        batch_size = st.sidebar.select_slider(
            "Batch Size",
            options=[16, 32, 64, 128],
            value=32
        )
    
        st.sidebar.markdown("---")
    
        # Répartition des données
        st.sidebar.subheader("📊 Répartition des Données")
    
        # Nouveau : Limite du nombre d'échantillons d'entraînement
        use_subset = st.sidebar.checkbox(
            "Limiter la taille du dataset d'entraînement",
            value=False,
            help="Permet de réduire le nombre d'échantillons pour observer l'impact de la taille du dataset"
        )
    
        max_train_samples = None
        if use_subset:
            max_train_samples = st.sidebar.slider(
                "Nombre max d'échantillons d'entraînement",
                min_value=50,
                max_value=1200,
                value=200,
                step=50,
                help="Réduire ce nombre permet d'observer l'underfitting dû au manque de données"
            )
        
            st.sidebar.info(f"""
            📉 **Réduction du dataset**
        
            Avec seulement **{max_train_samples}** échantillons d'entraînement :
            - Moins de données pour apprendre
            - Risque d'underfitting augmenté
            - Patterns moins bien capturés
        
            **Objectif :** Observer l'impact de la quantité de données sur l'apprentissage.
            """)
    
        test_size = st.sidebar.slider(
            "Taille du set de test (%)",
            min_value=10,
            max_value=40,
            value=20,
            step=5
        )
    
        val_size = st.sidebar.slider(
            "Taille du set de validation (%)",
            min_value=10,
            max_value=30,
            value=15,
            step=5
        )
    
        # ========================================================================
        # BOUTON D'ENTRAÎNEMENT
        # ========================================================================
    
        st.sidebar.markdown("---")
    
        if st.sidebar.button("🚀 Lancer l'Entraînement", type="primary"):
        
            # Préparation des données
            with st.spinner("Préparation des données..."):
                # Split train/temp
                X_train, X_temp, y_train, y_temp = train_test_split(
                    X, y, test_size=(test_size + val_size) / 100, 
                    stratify=y, random_state=SEED
                )
            
                # Split temp en val/test
                val_ratio = val_size / (test_size + val_size)
                X_val, X_test, y_val, y_test = train_test_split(
                    X_temp, y_temp, test_size=1-val_ratio, 
                    stratify=y_temp, random_state=SEED
                )
            
                # Réduction du dataset d'entraînement si demandée
                if max_train_samples is not None and len(X_train) > max_train_samples:
                    original_train_size = len(X_train)
                    # Utiliser train_test_split pour garder la stratification
                    X_train, _, y_train, _ = train_test_split(
                        X_train, y_train,
                        train_size=max_train_samples,
                        stratify=y_train,
                        random_state=SEED
                    )
                    st.warning(f"⚠️ Dataset réduit : {original_train_size} → {len(X_train)} échantillons d'entraînement")
            
                # Normalisation
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                X_test_scaled = scaler.transform(X_test)
            
                # Conversion en tenseurs
                train_dataset = TensorDataset(
                    torch.FloatTensor(X_train_scaled),
                    torch.LongTensor(y_train)
                )
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val_scaled),
                    torch.LongTensor(y_val)
                )
                test_dataset = TensorDataset(
                    torch.FloatTensor(X_test_scaled),
                    torch.LongTensor(y_test)
                )
            
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
            # Message de confirmation
            if max_train_samples is not None and len(X_train) < 1000:
                ratio = len(X_train) / 1200  # Approximativement le nombre total d'échantillons train normal
                st.info(f"""
                📊 **Configuration avec données limitées**
            
                - Train : **{len(X_train)}** échantillons ({ratio*100:.0f}% du dataset complet)
                - Val : {len(X_val)} échantillons
                - Test : {len(X_test)} échantillons
            
                Avec peu de données, le modèle peut :
                - Avoir du mal à apprendre (underfitting)
                - Être plus sensible au bruit
                - Nécessiter une architecture plus simple
                """)
            else:
                st.success(f"✓ Données préparées : Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
        
            # Création du modèle
            model = SpectralClassifier(input_dim=2090, hidden_dims=hidden_dims).to(device)
        
            # Configuration de l'entraînement
            class_counts = np.bincount(y_train)
            class_weights = 1.0 / class_counts
            class_weights = class_weights / class_weights.sum()
            class_weights_t = torch.FloatTensor(class_weights).to(device)
        
            criterion = nn.CrossEntropyLoss(weight=class_weights_t)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
            # Historique
            history = {
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
        
            # Conteneurs pour les graphiques
            col1, col2 = st.columns(2)
        
            with col1:
                st.subheader("📉 Évolution de la Loss")
                loss_chart = st.empty()
        
            with col2:
                st.subheader("📈 Évolution de l'Accuracy")
                acc_chart = st.empty()
        
            progress_bar = st.progress(0)
            status_text = st.empty()
        
            # Boucle d'entraînement
            for epoch in range(num_epochs):
                # Train
                train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
            
                # Validation
                val_loss, val_acc, _, _, _ = validate(model, val_loader, criterion, device)
            
                # Historique
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
                # Mise à jour des graphiques
                if (epoch + 1) % 2 == 0 or epoch == num_epochs - 1:
                    # Graphique Loss
                    fig_loss, ax_loss = plt.subplots(figsize=(8, 4))
                    ax_loss.plot(history['train_loss'], label='Train Loss', linewidth=2)
                    ax_loss.plot(history['val_loss'], label='Val Loss', linewidth=2)
                    ax_loss.set_xlabel('Epoch')
                    ax_loss.set_ylabel('Loss')
                    ax_loss.set_title('Évolution de la Loss')
                    ax_loss.legend()
                    ax_loss.grid(True, alpha=0.3)
                    loss_chart.pyplot(fig_loss)
                    plt.close()
                
                    # Graphique Accuracy
                    fig_acc, ax_acc = plt.subplots(figsize=(8, 4))
                    ax_acc.plot(history['train_acc'], label='Train Acc', linewidth=2)
                    ax_acc.plot(history['val_acc'], label='Val Acc', linewidth=2)
                    ax_acc.set_xlabel('Epoch')
                    ax_acc.set_ylabel('Accuracy (%)')
                    ax_acc.set_title('Évolution de l\'Accuracy')
                    ax_acc.legend()
                    ax_acc.grid(True, alpha=0.3)
                    acc_chart.pyplot(fig_acc)
                    plt.close()
            
                # Mise à jour de la barre de progression
                progress_bar.progress((epoch + 1) / num_epochs)
                status_text.text(f"Epoch {epoch+1}/{num_epochs} | "
                               f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                               f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
            st.success("✅ Entraînement terminé !")
        
            # ====================================================================
            # ÉVALUATION FINALE SUR LE SET DE TEST
            # ====================================================================
        
            st.markdown("---")
            st.header("📊 Évaluation sur le Set de Test")
        
            test_loss, test_acc, test_preds, test_labels, test_probs = validate(
                model, test_loader, criterion, device
            )
        
            # Métriques
            col1, col2, col3, col4 = st.columns(4)
        
            with col1:
                st.metric("Accuracy", f"{test_acc:.2f}%")
        
            with col2:
                precision = precision_score(test_labels, test_preds)
                st.metric("Precision", f"{precision:.3f}")
        
            with col3:
                recall = recall_score(test_labels, test_preds)
                st.metric("Recall", f"{recall:.3f}")
        
            with col4:
                f1 = f1_score(test_labels, test_preds)
                st.metric("F1-Score", f"{f1:.3f}")
        
            # Visualisations
            col1, col2 = st.columns(2)
        
            with col1:
                st.subheader("Matrice de Confusion")
                cm = confusion_matrix(test_labels, test_preds)
                fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                           xticklabels=['Normal', 'Cancer'],
                           yticklabels=['Normal', 'Cancer'],
                           ax=ax_cm)
                ax_cm.set_ylabel('Vrai Label')
                ax_cm.set_xlabel('Prédiction')
                ax_cm.set_title('Matrice de Confusion')
                st.pyplot(fig_cm)
                plt.close()
        
            with col2:
                st.subheader("Courbe ROC")
                fpr, tpr, _ = roc_curve(test_labels, test_probs)
                auc = roc_auc_score(test_labels, test_probs)
            
                fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                ax_roc.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
                ax_roc.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Aléatoire')
                ax_roc.set_xlabel('Taux de Faux Positifs')
                ax_roc.set_ylabel('Taux de Vrais Positifs')
                ax_roc.set_title('Courbe ROC')
                ax_roc.legend()
                ax_roc.grid(True, alpha=0.3)
                st.pyplot(fig_roc)
                plt.close()
        
            # Métriques détaillées
            st.markdown("---")
            st.subheader("📋 Rapport de Classification Détaillé")
        
            col1, col2 = st.columns(2)
        
            with col1:
                st.markdown("**Classe Normal (0)**")
                tn, fp, fn, tp = cm.ravel()
                st.write(f"- Vrais Négatifs (TN): {tn}")
                st.write(f"- Faux Positifs (FP): {fp}")
                st.write(f"- Spécificité: {tn/(tn+fp):.3f}")
        
            with col2:
                st.markdown("**Classe Cancer (1)**")
                st.write(f"- Vrais Positifs (TP): {tp}")
                st.write(f"- Faux Négatifs (FN): {fn}")
                st.write(f"- Sensibilité: {tp/(tp+fn):.3f}")
        
            # Informations pédagogiques
            st.markdown("---")
            st.info("""
            ### 📖 Interprétation des Résultats
        
            - **Accuracy** : Proportion de prédictions correctes (toutes classes confondues)
            - **Precision** : Parmi les échantillons prédits comme cancer, quelle proportion l'est vraiment ?
            - **Recall (Sensibilité)** : Parmi les vrais cancers, quelle proportion est détectée ?
            - **F1-Score** : Moyenne harmonique de la précision et du recall
            - **AUC** : Aire sous la courbe ROC, mesure la capacité du modèle à discriminer les classes
        
            **Interprétation médicale** :
            - Un **Recall élevé** est crucial pour ne pas manquer de cas de cancer (minimiser les faux négatifs)
            - Une **Precision élevée** évite les diagnostics erronés de cancer (minimiser les faux positifs)
            """)

    # ============================================================================
    # POINT D'ENTRÉE
    # ============================================================================

if __name__ == "__main__":
    main()
