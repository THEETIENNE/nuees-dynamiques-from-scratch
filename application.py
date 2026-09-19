"""Interface Streamlit pour choisir la représentation des nuées."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

RACINE = Path(__file__).resolve().parent
sys.path.insert(0, str(RACINE / "src"))
from nuees_dynamiques_generales import NueesDynamiquesGenerales

st.set_page_config(page_title="Nuées dynamiques", page_icon="☁️", layout="wide")
st.title("☁️ Segmentation par Nuées Dynamiques")
st.caption("Cadre général de Diday : la représentation d'une classe est choisie par l'utilisateur.")

fichier = st.file_uploader("Dataset CSV ou Excel", type=["csv", "xlsx", "xls"])
if fichier is None:
    chemin = RACINE / "data" / "Iris_Dataset_Segmentation.xlsx"
    df = pd.read_excel(chemin, sheet_name="Donnees")
    st.info("Dataset Iris chargé par défaut : 150 fleurs et 4 mesures numériques.")
else:
    df = pd.read_csv(fichier) if fichier.name.lower().endswith(".csv") else pd.read_excel(fichier)

numeriques = list(df.select_dtypes(include=np.number).columns)
variables = st.multiselect("Variables numériques utilisées", numeriques, default=numeriques[:min(5, len(numeriques))])
c1, c2 = st.columns(2)
with c1:
    k = st.number_input("Nombre de classes", 2, 20, 3)
    libelles = {
        "Un point (k-means)": "centroide",
        "Plusieurs points représentatifs": "points",
        "Axes factoriels": "axes",
        "Distribution gaussienne": "distribution",
    }
    choix = st.selectbox("Représentation de chaque nuée", list(libelles))
with c2:
    n_rep = st.number_input("Points représentatifs par nuée", 2, 10, 3,
                            disabled=libelles[choix] != "points")
    dim_axes = st.number_input("Axes par nuée", 1, 5, 1,
                               disabled=libelles[choix] != "axes")

if st.button("Segmenter les données", type="primary"):
    if not variables:
        st.error("Sélectionnez au moins une variable numérique.")
    else:
        propre = df.dropna(subset=variables).copy()
        X = propre[variables].to_numpy(float)
        ecarts = X.std(axis=0); ecarts[ecarts == 0] = 1
        Xn = (X - X.mean(axis=0)) / ecarts
        modele = NueesDynamiquesGenerales(
            n_clusters=k, representation=libelles[choix], n_representants=n_rep,
            dimension_axes=min(dim_axes, max(1, Xn.shape[1] - 1)), n_init=10,
            random_state=42
        ).fit(Xn)
        propre["nuee"] = modele.labels_ + 1
        st.success(f"Segmentation terminée : {k} nuées, {modele.n_iter_} itérations, critère = {modele.criterion_:.4f}")
        st.dataframe(propre.head(100), use_container_width=True)
        st.download_button("Télécharger les données segmentées", propre.to_csv(index=False).encode(),
                           "donnees_segmentees.csv", "text/csv")
        g1, g2 = st.columns(2)
        with g1:
            fig, ax = plt.subplots()
            if Xn.shape[1] >= 2:
                u, _, _ = np.linalg.svd(Xn - Xn.mean(0), full_matrices=False)
                coord = u[:, :2]
            else:
                coord = np.column_stack([Xn[:, 0], np.zeros(len(Xn))])
            ax.scatter(coord[:, 0], coord[:, 1], c=modele.labels_, cmap="viridis", s=22)
            ax.set_title("Projection 2D des nuées"); ax.set_xlabel("Dimension 1"); ax.set_ylabel("Dimension 2")
            st.pyplot(fig)
        with g2:
            fig, ax = plt.subplots()
            ax.plot(range(1, len(modele.history_) + 1), modele.history_, "o-")
            ax.set_title("Convergence"); ax.set_xlabel("Itération"); ax.set_ylabel("Critère d'adéquation")
            st.pyplot(fig)
