# train.py
"""
Script principal d'entraînement : recherche aléatoire d'hyperparamètres
(dropout, régularisation L2, learning rate, batch size, profondeur du
réseau...), entraînement et conservation des 3 meilleurs modèles selon
l'accuracy en validation.
"""

import argparse
import itertools
import random
import numpy as np
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from preprocessing import load_data_categorical
from model import create_model
from evaluate import plot_history, evaluate_model

# ============================================================
# 📂 Chemins vers les données
# Exemple d'organisation des dossiers :
#   ./data/
#      ├── Train/
#      ├── Validation/
#      └── Tests/
#
# Lors de l'exécution du script, vous devez préciser :
#   --train_dir ./data/Train
#   --val_dir   ./data/Validation
#   --test_dir  ./data/Tests
#
# Exemple de commande :
#   python train.py --train_dir ./data/Train \
#                   --val_dir ./data/Validation \
#                   --test_dir ./data/Tests \
#                   --epochs 50 --n_samples 10
# ============================================================



def main(args):
    # Charger les données
    X_train, y_train, y_train_int, X_val, y_val, y_val_int, X_test, y_test, y_test_int, class_names = load_data_categorical(
        args.train_dir, args.val_dir, args.test_dir
    )

    # Définition de la grille d'hyperparamètres
    param_grid = {
        "dropout_rate": [0.15, 0.2, 0.25, 0.3],
        "l2_val": [1e-6, 1e-5, 1e-4],
        "lr": [1e-2, 1e-3, 1e-4],
        "batch_size": [8, 16, 32],
        "reduce_factor": [0.1, 0.2, 0.5],
        "reduce_patience": [3, 5, 7],
        "n_conv_layers": [2, 3, 4],
        "n_dense_layers": [2, 3, 4]
    }

    keys, values = zip(*param_grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    # Tirage aléatoire de combinaisons
    random_combos = random.sample(all_combinations, args.n_samples)

    top_models = []

    for idx, params in enumerate(random_combos, 1):
        print(f"\n===== Expérience {idx}/{args.n_samples} – params: {params}")

        X_tr, y_tr = shuffle(X_train, y_train, random_state=42)
        X_v, y_v = shuffle(X_val, y_val, random_state=42)

        model = create_model(
            input_shape=X_train.shape[1:],
            num_classes=len(class_names),
            dropout_rate=params['dropout_rate'],
            l2_val=params['l2_val'],
            lr=params['lr'],
            n_conv_layers=params['n_conv_layers'],
            n_dense_layers=params['n_dense_layers']
        )

        callbacks = [
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=params['reduce_factor'],
                patience=params['reduce_patience'],
                min_lr=1e-8,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            )
        ]

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_v, y_v),
            epochs=args.epochs,
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=0
        )

        best_epoch = np.argmax(history.history['val_accuracy'])
        best_val_acc = history.history['val_accuracy'][best_epoch]
        print(f" → Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

        # Ajout au classement
        top_models.append({
            "val_acc": best_val_acc,
            "params": params,
            "history": history,
            "best_epoch": best_epoch,
            "model": model
        })

        # On garde seulement les 3 meilleurs
        top_models = sorted(top_models, key=lambda x: x['val_acc'], reverse=True)[:3]

    # Affichage des 3 meilleurs modèles
    for rank, info in enumerate(top_models, 1):
        print(f"\n===== TOP {rank} – Validation Accuracy: {info['val_acc']:.4f}")
        print("Params:", info['params'])
        plot_history(info["history"], info["best_epoch"])
        evaluate_model(info["model"], X_test, y_test, y_test_int, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, required=True, help="Dossier train")
    parser.add_argument("--val_dir", type=str, required=True, help="Dossier validation")
    parser.add_argument("--test_dir", type=str, required=True, help="Dossier test")
    parser.add_argument("--epochs", type=int, default=50, help="Nombre d'époques d'entraînement")
    parser.add_argument("--n_samples", type=int, default=200, help="Nombre de combinaisons d'hyperparamètres testées")
    args = parser.parse_args()

    main(args)
