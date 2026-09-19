"""Algorithme des nuées dynamiques (k-means) implémenté from scratch.

Seul NumPy est utilisé pour les opérations numériques. Aucun algorithme de
clustering externe n'intervient dans l'apprentissage ou la prédiction.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class ResultatExecution:
    centres: np.ndarray
    etiquettes: np.ndarray
    inertie: float
    historique: list[float]
    n_iterations: int


class NueesDynamiques:
    """Partitionne des observations en k classes par minimisation de l'inertie."""

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4,
                 n_init=10, random_state=None):
        if n_clusters < 1 or max_iter < 1 or n_init < 1 or tol < 0:
            raise ValueError("Paramètres invalides.")
        self.n_clusters = int(n_clusters)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.n_init = int(n_init)
        self.random_state = random_state

    @staticmethod
    def _verifier_X(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
            raise ValueError("X doit être une matrice numérique non vide.")
        if not np.isfinite(X).all():
            raise ValueError("X contient une valeur NaN ou infinie.")
        return X

    @staticmethod
    def _distances_carre(X, centres):
        # Matrice (n_observations, k), sans appel à une fonction de distance.
        differences = X[:, np.newaxis, :] - centres[np.newaxis, :, :]
        return np.sum(differences * differences, axis=2)

    def _initialiser_centres(self, X, rng):
        indices = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        return X[indices].copy()

    def _recalculer_centres(self, X, etiquettes, anciens_centres, distances):
        nouveaux = np.empty_like(anciens_centres)
        # Les classes vides sont replacées sur les observations les plus mal
        # représentées, ce qui évite NaN et favorise une meilleure partition.
        erreurs = distances[np.arange(X.shape[0]), etiquettes]
        candidats = list(np.argsort(erreurs)[::-1])
        utilises = set()
        for j in range(self.n_clusters):
            membres = X[etiquettes == j]
            if len(membres):
                nouveaux[j] = np.mean(membres, axis=0)
            else:
                idx = next(i for i in candidats if i not in utilises)
                utilises.add(idx)
                nouveaux[j] = X[idx]
        return nouveaux

    def _une_execution(self, X, rng):
        centres = self._initialiser_centres(X, rng)
        distances_initiales = self._distances_carre(X, centres)
        etiquettes_initiales = np.argmin(distances_initiales, axis=1)
        historique = [float(np.sum(
            distances_initiales[np.arange(X.shape[0]), etiquettes_initiales]
        ))]
        etiquettes_precedentes = None

        for iteration in range(1, self.max_iter + 1):
            distances = self._distances_carre(X, centres)
            etiquettes = np.argmin(distances, axis=1)
            nouveaux_centres = self._recalculer_centres(
                X, etiquettes, centres, distances
            )

            # Réaffectation après déplacement : l'inertie enregistrée correspond
            # toujours aux centres et étiquettes retournés à cette itération.
            nouvelles_distances = self._distances_carre(X, nouveaux_centres)
            nouvelles_etiquettes = np.argmin(nouvelles_distances, axis=1)
            inertie = float(np.sum(
                nouvelles_distances[np.arange(X.shape[0]), nouvelles_etiquettes]
            ))
            historique.append(inertie)

            deplacement = float(np.linalg.norm(nouveaux_centres - centres))
            partition_stable = (
                etiquettes_precedentes is not None
                and np.array_equal(nouvelles_etiquettes, etiquettes_precedentes)
            )
            centres = nouveaux_centres
            etiquettes_precedentes = nouvelles_etiquettes.copy()
            if deplacement <= self.tol or partition_stable:
                break

        distances = self._distances_carre(X, centres)
        etiquettes = np.argmin(distances, axis=1)
        inertie = float(np.sum(distances[np.arange(X.shape[0]), etiquettes]))
        historique[-1] = inertie
        return ResultatExecution(
            centres, etiquettes, inertie, historique, iteration
        )

    def fit(self, X):
        X = self._verifier_X(X)
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters ne peut pas dépasser le nombre de points.")
        rng = np.random.default_rng(self.random_state)
        meilleur = min(
            (self._une_execution(X, rng) for _ in range(self.n_init)),
            key=lambda resultat: resultat.inertie,
        )
        self.cluster_centers_ = meilleur.centres
        self.labels_ = meilleur.etiquettes
        self.inertia_ = meilleur.inertie
        self.inertia_history_ = meilleur.historique
        self.n_iter_ = meilleur.n_iterations
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        if not hasattr(self, "cluster_centers_"):
            raise RuntimeError("Le modèle doit être entraîné avec fit avant predict.")
        X = self._verifier_X(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Nombre de variables incompatible avec l'entraînement.")
        return np.argmin(self._distances_carre(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
