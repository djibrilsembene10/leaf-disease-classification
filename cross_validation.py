# cross_validation.py
"""
Validation croisée stratifiée (k-fold) du modèle de classification de
maladies foliaires : pondération des classes, sauvegarde d'un modèle par
fold, calcul des métriques (accuracy, F1-score) sur validation et test,
matrices de confusion par fold et globales.

Réutilise l'architecture définie dans model.py (create_model) afin de ne
pas dupliquer la définition du CNN.

Exemple d'utilisation :
    python cross_validation.py --train_dir data/Train --val_dir data/Validation \
        --test_dir data/Tests --k_folds 5 --epochs 1000 --batch_size 32 \
        --run_name reelles
"""

import argparse
import json
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import to_categorical

from model import create_model


# ------------------------------------------------------------------
# Chargement des données (conserve les noms de fichiers du jeu de test,
# nécessaires pour l'annotation des prédictions dans predict_and_visualize.py)
# ------------------------------------------------------------------
def load_data_from_dirs(train_dir, val_dir, test_dir, img_size=(128, 128)):
    """Charge les images depuis des répertoires train/val/test organisés au
    format un-sous-dossier-par-classe, et renvoie en plus les noms de
    fichiers du jeu de test.
    """
    class_names = sorted([
        d for d in os.listdir(train_dir)
        if not d.startswith(".") and os.path.isdir(os.path.join(train_dir, d))
    ])
    class_dict = {name: idx for idx, name in enumerate(class_names)}

    def _collect_from(dir_path):
        imgs, labels, filenames = [], [], []
        for cname in class_names:
            folder = os.path.join(dir_path, cname)
            for fname in os.listdir(folder):
                path = os.path.join(folder, fname)
                img = cv2.imread(path)
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                imgs.append(img)
                labels.append(class_dict[cname])
                filenames.append(fname)
        X = np.array(imgs, dtype=np.float32) / 255.0
        y_int = np.array(labels, dtype=np.int32)
        return X, y_int, filenames

    X_train, y_train_int, _ = _collect_from(train_dir)
    X_val, y_val_int, _ = _collect_from(val_dir)
    X_test, y_test_int, fn_test = _collect_from(test_dir)

    return X_train, y_train_int, X_val, y_val_int, X_test, y_test_int, class_names, fn_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation croisée k-fold du classifieur de maladies foliaires")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--k_folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout_rate", type=float, default=0.15)
    parser.add_argument("--l2_val", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--reduce_factor", type=float, default=0.2)
    parser.add_argument("--reduce_patience", type=int, default=2)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_folds")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_name", type=str, default="reelles",
                         help="Suffixe pour différencier les runs (ex: 'reelles', 'simulees')")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _plot_fold_curves(history, fold, results_dir, run_name):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.title(f"Fold {fold} - Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.title(f"Fold {fold} - Accuracy")
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(results_dir, f"curves_fold_{run_name}_{fold}.png")
    plt.savefig(save_path)
    plt.close()


def _plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title(title)
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    if save_path:
        plt.savefig(save_path)
    plt.close()


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    X_train, y_train_int, X_val, y_val_int, X_test, y_test_int, class_names, fn_test = load_data_from_dirs(
        args.train_dir, args.val_dir, args.test_dir
    )
    y_train_cat = to_categorical(y_train_int, num_classes=len(class_names))
    y_val_cat = to_categorical(y_val_int, num_classes=len(class_names))
    y_test_cat = to_categorical(y_test_int, num_classes=len(class_names))

    skf_train = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    skf_val = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)
    skf_test = StratifiedKFold(n_splits=args.k_folds, shuffle=True, random_state=args.seed)

    acc_list_val, f1_list_val = [], []
    acc_list_test, f1_list_test = [], []
    all_y_true_val, all_y_pred_val = [], []
    all_y_true_test, all_y_pred_test = [], []

    folds = zip(
        skf_train.split(X_train, y_train_int),
        skf_val.split(X_val, y_val_int),
        skf_test.split(X_test, y_test_int),
    )

    for fold, ((train_idx, _), (val_idx, _), (test_idx, _)) in enumerate(folds, start=1):
        print(f"\n--- Fold {fold}/{args.k_folds} ---")

        X_tr, y_tr = X_train[train_idx], y_train_cat[train_idx]
        X_vl, y_vl = X_val[val_idx], y_val_cat[val_idx]
        X_te, y_te = X_test[test_idx], y_test_cat[test_idx]

        # Pondération des classes pour compenser un éventuel déséquilibre du dataset
        cw = class_weight.compute_class_weight(
            "balanced", classes=np.arange(len(class_names)), y=np.argmax(y_tr, axis=1)
        )
        class_weights = dict(enumerate(cw))

        model = create_model(
            input_shape=X_train.shape[1:],
            num_classes=len(class_names),
            dropout_rate=args.dropout_rate,
            l2_val=args.l2_val,
            lr=args.lr,
        )

        callbacks = [
            ReduceLROnPlateau(monitor="val_loss", factor=args.reduce_factor,
                               patience=args.reduce_patience, min_lr=1e-8, verbose=1),
            EarlyStopping(monitor="val_loss", patience=args.early_stopping_patience,
                           restore_best_weights=True, verbose=1),
        ]

        history = model.fit(
            X_tr, y_tr,
            validation_data=(X_vl, y_vl),
            epochs=args.epochs,
            batch_size=args.batch_size,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=1,
        )

        fold_model_path = os.path.join(args.checkpoint_dir, f"model_fold_{args.run_name}_{fold}.h5")
        model.save(fold_model_path)
        print(f"Modèle du fold {fold} sauvegardé dans : {fold_model_path}")

        _plot_fold_curves(history, fold, args.results_dir, args.run_name)

        # Évaluation validation
        y_val_pred = np.argmax(model.predict(X_vl), axis=1)
        y_val_true = np.argmax(y_vl, axis=1)
        acc_val = accuracy_score(y_val_true, y_val_pred)
        f1_val = f1_score(y_val_true, y_val_pred, average="macro")
        acc_list_val.append(acc_val)
        f1_list_val.append(f1_val)
        all_y_true_val.extend(y_val_true)
        all_y_pred_val.extend(y_val_pred)
        print(f"Fold {fold} - Val Accuracy: {acc_val:.4f}, F1: {f1_val:.4f}")

        # Évaluation test
        y_test_pred = np.argmax(model.predict(X_te), axis=1)
        y_test_true = np.argmax(y_te, axis=1)
        acc_test = accuracy_score(y_test_true, y_test_pred)
        f1_test = f1_score(y_test_true, y_test_pred, average="macro")
        acc_list_test.append(acc_test)
        f1_list_test.append(f1_test)
        all_y_true_test.extend(y_test_true)
        all_y_pred_test.extend(y_test_pred)
        print(f"Fold {fold} - Test Accuracy: {acc_test:.4f}, F1: {f1_test:.4f}")

        print(f"\n--- Rapport de classification Fold {fold} (Test) ---")
        print(classification_report(y_test_true, y_test_pred, target_names=class_names))

        _plot_confusion_matrix(
            y_test_true, y_test_pred, class_names,
            title=f"Matrice de confusion - Fold {fold} (Test)",
            save_path=os.path.join(args.results_dir, f"cm_fold_{args.run_name}_{fold}.png"),
        )

    # ------------------- Résultats agrégés sur l'ensemble des folds -------------------
    print("\n--- Résultats Validation (CV) ---")
    print(f"Accuracy CV : {np.mean(acc_list_val):.4f} ± {np.std(acc_list_val):.4f}")
    print(f"F1-score CV : {np.mean(f1_list_val):.4f} ± {np.std(f1_list_val):.4f}")

    print("\n--- Résultats Test (moyenne sur les folds) ---")
    print(f"Accuracy Test : {np.mean(acc_list_test):.4f} ± {np.std(acc_list_test):.4f}")
    print(f"F1-score Test : {np.mean(f1_list_test):.4f} ± {np.std(f1_list_test):.4f}")

    _plot_confusion_matrix(
        all_y_true_val, all_y_pred_val, class_names,
        title="Matrice de confusion Validation (tous folds)",
        save_path=os.path.join(args.results_dir, f"cm_val_global_{args.run_name}.png"),
    )
    _plot_confusion_matrix(
        all_y_true_test, all_y_pred_test, class_names,
        title="Matrice de confusion Test (tous folds)",
        save_path=os.path.join(args.results_dir, f"cm_test_global_{args.run_name}.png"),
    )

    print("\n--- Rapport de classification (Test, tous folds) ---")
    print(classification_report(all_y_true_test, all_y_pred_test, target_names=class_names))

    # Sauvegarde du résumé chiffré + des classes (utile pour predict_and_visualize.py)
    summary = {
        "class_names": class_names,
        "k_folds": args.k_folds,
        "val_accuracy_mean": float(np.mean(acc_list_val)),
        "val_accuracy_std": float(np.std(acc_list_val)),
        "val_f1_mean": float(np.mean(f1_list_val)),
        "val_f1_std": float(np.std(f1_list_val)),
        "test_accuracy_mean": float(np.mean(acc_list_test)),
        "test_accuracy_std": float(np.std(acc_list_test)),
        "test_f1_mean": float(np.mean(f1_list_test)),
        "test_f1_std": float(np.std(f1_list_test)),
    }
    results_path = os.path.join(args.results_dir, f"cv_summary_{args.run_name}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nRésumé sauvegardé dans : {results_path}")

    class_names_path = os.path.join(args.checkpoint_dir, "class_names.json")
    with open(class_names_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f)
    print(f"Classes sauvegardées dans : {class_names_path}")


if __name__ == "__main__":
    main()
