"""Étude de segmentation du dataset Iris avec les représentations de Diday."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from nuees_dynamiques_generales import NueesDynamiquesGenerales  # noqa: E402


def main():
    iris = load_iris(as_frame=True)
    X = iris.data.to_numpy(float)
    y = iris.target.to_numpy()
    Xn = StandardScaler().fit_transform(X)
    modes = ["centroide", "points", "axes", "distribution"]
    noms = ["Centroïde", "3 points", "1 axe", "Gaussienne"]
    resultats = []
    modeles = {}
    for mode, nom in zip(modes, noms):
        modele = NueesDynamiquesGenerales(
            3, representation=mode, n_representants=3, dimension_axes=1,
            n_init=20, random_state=42
        ).fit(Xn)
        modeles[mode] = modele
        resultats.append({
            "representation": nom,
            "silhouette": silhouette_score(Xn, modele.labels_),
            "ari": adjusted_rand_score(y, modele.labels_),
            "iterations": modele.n_iter_,
            "critere": modele.criterion_,
        })

    # Le meilleur accord avec les trois espèces est obtenu par la distribution.
    meilleur = modeles["distribution"]
    sortie = iris.frame.drop(columns="target").copy()
    sortie.columns = ["sepal_length_cm", "sepal_width_cm", "petal_length_cm", "petal_width_cm"]
    sortie["species"] = [iris.target_names[i] for i in y]
    sortie["nuee_predite"] = meilleur.labels_ + 1
    sortie.to_csv(RACINE / "experiments" / "iris_segmentation_resultats.csv", index=False)

    coord = PCA(n_components=2).fit_transform(Xn)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].scatter(coord[:, 0], coord[:, 1], c=meilleur.labels_, cmap="viridis", s=34)
    axes[0].set_title("Nuées trouvées (distribution, k = 3)")
    axes[1].scatter(coord[:, 0], coord[:, 1], c=y, cmap="viridis", s=34)
    axes[1].set_title("Espèces réelles (évaluation seulement)")
    for ax in axes:
        ax.set_xlabel("Composante principale 1"); ax.set_ylabel("Composante principale 2")
        ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(RACINE / "figures" / "iris_segmentation.png", dpi=180)
    plt.close(fig)

    x = np.arange(len(noms)); largeur = .36
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.bar(x-largeur/2, [r["ari"] for r in resultats], largeur, label="ARI", color="#145da0")
    ax.bar(x+largeur/2, [r["silhouette"] for r in resultats], largeur, label="Silhouette", color="#2a9d8f")
    ax.set_xticks(x, noms); ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Comparaison sur Iris pour k = 3"); ax.legend(); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); fig.savefig(RACINE / "figures" / "iris_comparaison.png", dpi=180)
    plt.close(fig)

    lignes = ["Dataset Iris : 150 observations, 4 variables, 3 espèces."]
    for r in resultats:
        lignes.append(f'{r["representation"]}: ARI={r["ari"]:.6f}, silhouette={r["silhouette"]:.6f}, '
                       f'itérations={r["iterations"]}, critère={r["critere"]:.6f}')
    texte = "\n".join(lignes) + "\n"
    (RACINE / "experiments" / "iris_resultats.txt").write_text(texte, encoding="utf-8")
    print(texte)


if __name__ == "__main__":
    main()
