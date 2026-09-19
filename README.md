# Nuées dynamiques from scratch

TP de Science des données de **KABONGO TSHIATA ETIENNE (M1 IA)**.

Le projet implémente l'algorithme des nuées dynamiques (k-means/Lloyd) sans
utiliser `sklearn.cluster.KMeans` dans le modèle principal. La comparaison avec
scikit-learn est facultative et intervient seulement après l'apprentissage.

## Installation et exécution

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\\Scripts\\activate
pip install -r requirements.txt
python experiments/experimentation.py
pytest -q
```

Les figures sont enregistrées dans `figures/` et les valeurs numériques dans
`experiments/resultats.txt`.

## Compilation du rapport

```bash
cd rapport
latexmk -pdf rapport.tex
```

## Structure

- `src/nuees_dynamiques.py` : implémentation complète ;
- `experiments/experimentation.py` : données, figures et validation ;
- `tests/` : tests automatisés ;
- `data/` : jeu de données généré avec une graine fixe ;
- `rapport/` : source LaTeX, bibliographie et PDF ;
- `figures/` : partition, convergence et méthode du coude.

## Dépôt GitHub

Après création d'un dépôt vide sur GitHub :

```bash
git init
git add .
git commit -m "TP nuées dynamiques from scratch"
git branch -M main
git remote add origin https://github.com/VOTRE-COMPTE/nuees-dynamiques-from-scratch.git
git push -u origin main
```

Remplacer ensuite `VOTRE-COMPTE` dans le rapport par le nom réel du compte.
