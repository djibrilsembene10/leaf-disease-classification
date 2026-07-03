# 🌿 Détection et Classification de Maladies Foliaires par Deep Learning

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Statut-En%20cours-yellow)

Pipeline **Deep Learning (TensorFlow/Keras)** pour la **classification** de maladies foliaires sur arbres fruitiers à partir d'images RGB, avec recherche d'hyperparamètres et validation croisée stratifiée.

> 🔒 **Note sur la confidentialité** : ce projet a été développé dans le cadre d'un CDD à l'INRAE Bordeaux Nouvelle-Aquitaine. Le dataset original ainsi qu'une partie du code sont hébergés sur un dépôt GitLab privé (INRAE) pour des raisons de confidentialité. Ce dépôt public présente la méthodologie, le pipeline et les résultats obtenus, sans les données propriétaires.

---

## 📌 Sommaire

- [Contexte](#-contexte)
- [Approche technique](#-approche-technique)
- [Structure du projet](#-structure-du-projet)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
  - [1. Recherche d'hyperparamètres](#1--recherche-dhyperparamètres-trainpy)
  - [2. Validation croisée](#2--validation-croisée-cross_validationpy)
  - [3. Visualisation des prédictions](#3--visualisation-des-prédictions-predict_and_visualizepy)
- [Résultats](#-résultats)
- [Stack technique](#-stack-technique)
- [Auteur](#-auteur)
- [Licence](#-licence)

---

## 🧩 Contexte

Les maladies foliaires impactent fortement les rendements agricoles. Ce projet vise à automatiser leur classification à partir d'images afin d'aider au diagnostic rapide et objectif sur le terrain.

Le projet a été initié lors d'un stage de recherche (2023) puis approfondi et industrialisé sur un CDD de 2 ans (2023–2025), en repartant de zéro sur la constitution du dataset et la pipeline afin d'en fiabiliser les résultats.

## 🧠 Approche technique

Le dataset est organisé au format `Train/`, `Validation/`, `Tests/`, chacun contenant un sous-dossier par classe :

```
data/
├── Train/
│   ├── sain/
│   └── malade/
├── Validation/
│   ├── sain/
│   └── malade/
└── Tests/
    ├── sain/
    └── malade/
```

Le pipeline comprend deux volets complémentaires :

1. **Recherche d'hyperparamètres** (`train.py`) : recherche aléatoire sur un large espace d'hyperparamètres (dropout, régularisation L2, learning rate, batch size, profondeur du réseau...), conservation des 3 meilleurs modèles selon l'accuracy en validation.
2. **Validation croisée stratifiée** (`cross_validation.py`) : évaluation plus robuste de la configuration retenue via un k-fold stratifié, avec pondération des classes pour compenser un éventuel déséquilibre du dataset.

Un script dédié (`predict_and_visualize.py`) permet également d'inspecter visuellement les erreurs du modèle (vrais/faux positifs/négatifs superposés en couleur sur les images de test).

## 📁 Structure du projet

```
.
├── data/                     # dataset (non inclus – données privées INRAE)
├── preprocessing.py           # chargement et prétraitement des images
├── model.py                   # architecture CNN configurable
├── train.py                   # recherche aléatoire d'hyperparamètres
├── cross_validation.py        # validation croisée stratifiée (k-fold)
├── evaluate.py                # courbes, matrice de confusion, ROC
├── predict_and_visualize.py   # annotation visuelle des prédictions
├── results/                  # figures, métriques, résumés JSON (générés)
├── checkpoints_folds/        # modèles sauvegardés par fold (générés)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## ⚙️ Installation

```bash
git clone https://github.com/djibrilsembene10/-leaf-disease-detection.git
cd -leaf-disease-detection
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate
pip install -r requirements.txt
```

## ▶️ Utilisation

### 1. 🔍 Recherche d'hyperparamètres (`train.py`)

```bash
python train.py --train_dir data/Train --val_dir data/Validation --test_dir data/Tests \
    --epochs 50 --n_samples 10
```

Teste `n_samples` combinaisons aléatoires d'hyperparamètres, affiche les courbes d'apprentissage et les métriques de test (accuracy, F1-score, matrice de confusion, courbes ROC) pour les 3 meilleurs modèles.

### 2. 📊 Validation croisée (`cross_validation.py`)

```bash
python cross_validation.py --train_dir data/Train --val_dir data/Validation --test_dir data/Tests \
    --k_folds 5 --epochs 1000 --batch_size 32 --run_name reelles
```

Entraîne un modèle par fold, sauvegarde chaque modèle (`checkpoints_folds/model_fold_<run_name>_<n>.h5`), calcule accuracy/F1 moyens ± écart-type sur validation et test, et exporte un résumé JSON dans `results/`.

### 3. 🖼️ Visualisation des prédictions (`predict_and_visualize.py`)

```bash
python predict_and_visualize.py --model_path checkpoints_folds/model_fold_reelles_3.h5 \
    --train_dir data/Train --val_dir data/Validation --test_dir data/Tests \
    --output_dir results/predictions_annotees
```

Génère, pour chaque image de test, une version annotée indiquant si la prédiction est un **vrai positif** (vert), **vrai négatif** (gris), **faux positif** (rouge) ou **faux négatif** (bleu).

## 📊 Résultats

| Métrique | Valeur |
|---|---|
| Précision | 87 % |
| F1-score | 87 % |

## 🛠️ Stack technique

`Python` · `TensorFlow / Keras` · `OpenCV` · `Scikit-learn` · `Matplotlib` · `Seaborn`

## 👤 Auteur

**Djibril Sembene**
Ingénieur Computer Vision & Deep Learning
📧 djibrilsembene10@gmail.com

## 📄 Licence

Ce projet est distribué sous licence MIT — voir le fichier [LICENSE](LICENSE) pour plus de détails.
Le dataset original reste la propriété de l'INRAE et n'est pas inclus dans ce dépôt.
