# Nuées dynamiques générales from scratch

TP de Science des données de **KABONGO TSHIATA ETIENNE (M1 IA)**.

Le projet implémente le cadre général des nuées dynamiques de Diday. Une classe
peut être représentée par un centroïde (cas particulier k-means), plusieurs
points représentatifs, des axes factoriels, une distribution gaussienne ou des
fonctions personnalisées. Aucun algorithme de clustering externe n'intervient
dans le modèle principal.

## Installation et exécution

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\\Scripts\\activate
pip install -r requirements.txt
python experiments/experimentation.py
python experiments/iris_experiment.py
pytest -q
streamlit run application.py
```

Les figures sont enregistrées dans `figures/` et les valeurs numériques dans
`experiments/resultats.txt`.

## Compilation du rapport

```bash
cd rapport
latexmk -pdf rapport.tex
```

## Structure

- `src/nuees_dynamiques.py` : cas particulier centroïde/k-means ;
- `src/nuees_dynamiques_generales.py` : cadre général multi-représentations ;
- `application.py` : interface de choix et import CSV/Excel ;
- `experiments/experimentation.py` : données, figures et validation ;
- `experiments/iris_experiment.py` : segmentation réelle du dataset Iris ;
- `data/Iris_Dataset_Segmentation.xlsx` : dataset documenté (150 fleurs) ;
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
