"""Expérience reproductible et production des figures du rapport."""

from pathlib import Path
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

RACINE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RACINE / "src"))
from nuees_dynamiques import NueesDynamiques  # noqa: E402


def creer_donnees(seed=42):
    rng = np.random.default_rng(seed)
    configurations = [((-3.0, -2.0), (0.75, 0.55)),
                      ((0.2, 3.2), (0.65, 0.80)),
                      ((3.6, -0.8), (0.85, 0.60))]
    groupes = [rng.normal(mu, sigma, size=(150, 2))
               for mu, sigma in configurations]
    return np.vstack(groupes), np.repeat(np.arange(3), 150)


def sauvegarder_csv(X, y):
    chemin = RACINE / "data" / "donnees_synthetiques.csv"
    chemin.parent.mkdir(exist_ok=True)
    with chemin.open("w", newline="", encoding="utf-8") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(["x1", "x2", "classe_reelle"])
        writer.writerows((*point, int(classe)) for point, classe in zip(X, y))


def coude(X):
    inerties = []
    for k in range(1, 9):
        modele = NueesDynamiques(k, n_init=15, random_state=42).fit(X)
        inerties.append(modele.inertia_)
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(range(1, 9), inerties, "o-", color="#145da0", linewidth=2)
    ax.axvline(3, color="#e76f51", linestyle="--", label="k retenu = 3")
    ax.set(xlabel="Nombre de classes k", ylabel="Inertie intra-classe",
           title="Méthode du coude")
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(RACINE / "figures" / "coude.png", dpi=180)
    plt.close(fig)
    return inerties


def main():
    (RACINE / "figures").mkdir(exist_ok=True)
    X, y_reel = creer_donnees()
    sauvegarder_csv(X, y_reel)
    modele = NueesDynamiques(3, max_iter=200, tol=1e-6,
                             n_init=20, random_state=42).fit(X)

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.scatter(X[:, 0], X[:, 1], c=modele.labels_, cmap="viridis",
               s=22, alpha=.78, edgecolors="none")
    ax.scatter(modele.cluster_centers_[:, 0], modele.cluster_centers_[:, 1],
               marker="X", s=210, c="#d62828", edgecolors="white",
               linewidths=1.3, label="Centres finaux")
    ax.set(xlabel="$x_1$", ylabel="$x_2$", title="Partition obtenue (k = 3)")
    ax.grid(alpha=.18); ax.legend(); fig.tight_layout()
    fig.savefig(RACINE / "figures" / "clusters.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    iterations = range(1, len(modele.inertia_history_) + 1)
    ax.plot(iterations, modele.inertia_history_, "o-", color="#2a9d8f", lw=2)
    ax.set(xlabel="Itération", ylabel="Inertie", title="Convergence de l'algorithme")
    ax.set_xticks(list(iterations)); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(RACINE / "figures" / "convergence.png", dpi=180)
    plt.close(fig)

    inerties = coude(X)

    # Validation seulement : sklearn n'est jamais utilisé dans notre classe.
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import adjusted_rand_score
        reference = KMeans(n_clusters=3, n_init=20, random_state=42).fit(X)
        ari_ref = adjusted_rand_score(reference.labels_, modele.labels_)
        ari_verite = adjusted_rand_score(y_reel, modele.labels_)
        comparaison = (f"Inertie sklearn : {reference.inertia_:.6f}\n"
                       f"ARI vs sklearn : {ari_ref:.6f}\n"
                       f"ARI vs classes génératrices : {ari_verite:.6f}\n")
    except ImportError:
        comparaison = "scikit-learn absent : comparaison facultative ignorée.\n"

    resume = (f"Observations : {len(X)}\nItérations : {modele.n_iter_}\n"
              f"Inertie finale : {modele.inertia_:.6f}\n"
              f"Centres :\n{modele.cluster_centers_}\n"
              f"Inerties k=1..8 : {[round(v, 3) for v in inerties]}\n"
              + comparaison)
    (RACINE / "experiments" / "resultats.txt").write_text(resume, encoding="utf-8")
    print(resume)


if __name__ == "__main__":
    main()
