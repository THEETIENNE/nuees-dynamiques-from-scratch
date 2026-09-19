"""Cadre général des Nuées Dynamiques inspiré de Diday (1971).

Le principe est séparé en deux opérateurs : représentation d'une classe et
adéquation d'un individu à cette représentation. Le k-means correspond au cas
``representation='centroide'``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any
import numpy as np


@dataclass
class ResultatNuees:
    etiquettes: np.ndarray
    representations: list[Any]
    critere: float
    historique: list[float]
    n_iterations: int


class NueesDynamiquesGenerales:
    MODES = {"centroide", "points", "axes", "distribution", "personnalisee"}

    def __init__(self, n_clusters=3, representation="centroide",
                 n_representants=3, dimension_axes=1, max_iter=100,
                 tol=1e-6, n_init=5, regularisation=1e-6,
                 fonction_representation: Callable | None = None,
                 fonction_adequation: Callable | None = None,
                 random_state=None):
        if representation not in self.MODES:
            raise ValueError(f"Mode inconnu. Choisir parmi {sorted(self.MODES)}")
        if n_clusters < 1 or n_representants < 1 or dimension_axes < 1:
            raise ValueError("Les paramètres de taille doivent être positifs.")
        if representation == "personnalisee" and (
            fonction_representation is None or fonction_adequation is None
        ):
            raise ValueError("Deux fonctions sont requises en mode personnalisé.")
        self.n_clusters, self.representation = int(n_clusters), representation
        self.n_representants, self.dimension_axes = int(n_representants), int(dimension_axes)
        self.max_iter, self.tol, self.n_init = int(max_iter), float(tol), int(n_init)
        self.regularisation, self.random_state = float(regularisation), random_state
        self.fonction_representation = fonction_representation
        self.fonction_adequation = fonction_adequation

    @staticmethod
    def _verifier(X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or not len(X) or not X.shape[1] or not np.isfinite(X).all():
            raise ValueError("X doit être une matrice numérique finie et non vide.")
        return X

    def _centroide(self, C, rng):
        return np.mean(C, axis=0)

    def _points(self, C, rng):
        m = min(self.n_representants, len(C))
        # Sélection déterministe et dispersée de prototypes appartenant à la classe.
        centre = C.mean(axis=0)
        indices = [int(np.argmin(np.sum((C - centre) ** 2, axis=1)))]
        while len(indices) < m:
            d = np.min(np.sum((C[:, None, :] - C[indices][None, :, :]) ** 2, axis=2), axis=1)
            d[indices] = -1
            indices.append(int(np.argmax(d)))
        prototypes = C[indices].copy()
        # Quelques mises à jour internes, puis projection sur des points réels (médoïdes).
        for _ in range(10):
            d = np.sum((C[:, None, :] - prototypes[None, :, :]) ** 2, axis=2)
            z = np.argmin(d, axis=1)
            nouveaux = prototypes.copy()
            for j in range(m):
                sous = C[z == j]
                if len(sous):
                    moyenne = sous.mean(axis=0)
                    nouveaux[j] = sous[np.argmin(np.sum((sous - moyenne) ** 2, axis=1))]
            if np.allclose(nouveaux, prototypes):
                break
            prototypes = nouveaux
        return prototypes

    def _axes(self, C, rng):
        centre = C.mean(axis=0)
        q = min(self.dimension_axes, C.shape[1] - 1, max(1, len(C) - 1))
        if C.shape[1] == 1:
            base = np.ones((1, 1))
        else:
            _, _, vt = np.linalg.svd(C - centre, full_matrices=False)
            base = vt[:q].T
        return {"centre": centre, "base": base}

    def _distribution(self, C, rng):
        moyenne = C.mean(axis=0)
        if len(C) <= 1:
            covariance = np.eye(C.shape[1]) * self.regularisation
        else:
            covariance = np.atleast_2d(np.cov(C, rowvar=False, bias=True))
            if covariance.shape != (C.shape[1], C.shape[1]):
                covariance = np.eye(C.shape[1]) * float(np.squeeze(covariance))
            covariance += np.eye(C.shape[1]) * self.regularisation
        return {"moyenne": moyenne, "covariance": covariance}

    def _representer(self, C, rng):
        if self.representation == "centroide": return self._centroide(C, rng)
        if self.representation == "points": return self._points(C, rng)
        if self.representation == "axes": return self._axes(C, rng)
        if self.representation == "distribution": return self._distribution(C, rng)
        return self.fonction_representation(C)

    def _adequation(self, X, rep):
        if self.representation == "centroide":
            return np.sum((X - rep) ** 2, axis=1)
        if self.representation == "points":
            return np.min(np.sum((X[:, None, :] - rep[None, :, :]) ** 2, axis=2), axis=1)
        if self.representation == "axes":
            centre, base = rep["centre"], rep["base"]
            ecarts = X - centre
            projection = (ecarts @ base) @ base.T
            return np.sum((ecarts - projection) ** 2, axis=1)
        if self.representation == "distribution":
            mu, cov = rep["moyenne"], rep["covariance"]
            inv = np.linalg.pinv(cov)
            signe, logdet = np.linalg.slogdet(cov)
            if signe <= 0: logdet = np.log(self.regularisation)
            ecarts = X - mu
            return .5 * (np.einsum("ij,jk,ik->i", ecarts, inv, ecarts) + logdet)
        return np.asarray(self.fonction_adequation(X, rep), dtype=float)

    def _une_execution(self, X, rng):
        # Initialisation dispersée : un premier point aléatoire, puis le point
        # le plus éloigné des représentants déjà choisis.
        indices = [int(rng.integers(len(X)))]
        while len(indices) < self.n_clusters:
            d = np.min(np.sum((X[:, None, :] - X[indices][None, :, :]) ** 2, axis=2), axis=1)
            d[indices] = -1
            indices.append(int(np.argmax(d)))
        z = np.argmin(np.sum((X[:, None, :] - X[indices][None, :, :]) ** 2, axis=2), axis=1)
        historique, reps = [], None
        for iteration in range(1, self.max_iter + 1):
            reps = [self._representer(X[z == j], rng) for j in range(self.n_clusters)]
            matrice = np.column_stack([self._adequation(X, r) for r in reps])
            nouveau_z = np.argmin(matrice, axis=1)
            # Répare les classes vides avec les individus les moins bien représentés.
            pertes = matrice[np.arange(len(X)), nouveau_z]
            for j in range(self.n_clusters):
                if not np.any(nouveau_z == j):
                    idx = int(np.argmax(pertes))
                    nouveau_z[idx], pertes[idx] = j, -np.inf
            critere = float(np.sum(matrice[np.arange(len(X)), nouveau_z]))
            historique.append(critere)
            stable = np.array_equal(z, nouveau_z)
            amelioration = abs(historique[-2] - critere) if len(historique) > 1 else np.inf
            z = nouveau_z
            if stable or amelioration <= self.tol:
                break
        reps = [self._representer(X[z == j], rng) for j in range(self.n_clusters)]
        matrice = np.column_stack([self._adequation(X, r) for r in reps])
        critere = float(np.sum(matrice[np.arange(len(X)), z]))
        return ResultatNuees(z, reps, critere, historique, iteration)

    def fit(self, X):
        X = self._verifier(X)
        if self.n_clusters > len(X): raise ValueError("k dépasse le nombre d'individus.")
        rng = np.random.default_rng(self.random_state)
        meilleur = min((self._une_execution(X, rng) for _ in range(self.n_init)),
                       key=lambda r: r.critere)
        self.labels_, self.representations_ = meilleur.etiquettes, meilleur.representations
        self.criterion_, self.history_, self.n_iter_ = meilleur.critere, meilleur.historique, meilleur.n_iterations
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        X = self._verifier(X)
        if not hasattr(self, "representations_"): raise RuntimeError("Exécuter fit avant predict.")
        return np.argmin(np.column_stack([self._adequation(X, r) for r in self.representations_]), axis=1)

    def fit_predict(self, X):
        return self.fit(X).labels_
