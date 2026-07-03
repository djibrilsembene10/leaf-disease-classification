# predict_and_visualize.py
"""
Charge un modèle entraîné (issu de cross_validation.py) et génère, pour
chaque image du jeu de test, une version annotée d'une superposition
colorée indiquant le type de prédiction — utile pour inspecter visuellement
les erreurs du modèle (classification binaire uniquement) :

    - Vert  : Vrai Positif
    - Gris  : Vrai Négatif
    - Rouge : Faux Positif
    - Bleu  : Faux Négatif

Exemple d'utilisation :
    python predict_and_visualize.py --model_path checkpoints_folds/model_fold_reelles_3.h5 \
        --train_dir data/Train --val_dir data/Validation --test_dir data/Tests \
        --output_dir results/predictions_annotees
"""

import argparse
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model

from cross_validation import load_data_from_dirs


# Couleurs au format BGR (convention OpenCV)
COLOR_FALSE_POSITIVE = (0, 0, 255)     # Rouge
COLOR_FALSE_NEGATIVE = (255, 0, 0)     # Bleu
COLOR_TRUE_POSITIVE = (0, 255, 0)      # Vert
COLOR_TRUE_NEGATIVE = (128, 128, 128)  # Gris


def annotate_and_save_predictions(model, X, y_true, filenames, output_dir, alpha=0.3):
    """Sauvegarde chaque image de test avec une superposition colorée
    indiquant si la prédiction est un vrai/faux positif/négatif.
    Suppose une classification binaire (classe positive = index 1).
    """
    os.makedirs(output_dir, exist_ok=True)

    pred_probs = model.predict(X)
    pred_classes = np.argmax(pred_probs, axis=1)

    for i in range(len(X)):
        img_uint8 = (X[i] * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

        pred, actual = pred_classes[i], y_true[i]
        overlay = np.zeros_like(img_bgr)

        if pred == 1 and actual == 0:
            overlay[:] = COLOR_FALSE_POSITIVE
        elif pred == 0 and actual == 1:
            overlay[:] = COLOR_FALSE_NEGATIVE
        elif pred == 1 and actual == 1:
            overlay[:] = COLOR_TRUE_POSITIVE
        elif pred == 0 and actual == 0:
            overlay[:] = COLOR_TRUE_NEGATIVE

        img_bgr = cv2.addWeighted(img_bgr, 1 - alpha, overlay, alpha, 0)
        cv2.imwrite(os.path.join(output_dir, filenames[i]), img_bgr)

    print(f"Toutes les images ont été sauvegardées dans : {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotation visuelle des prédictions du modèle sur le jeu de test")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le modèle .h5 à charger")
    parser.add_argument("--train_dir", type=str, required=True)
    parser.add_argument("--val_dir", type=str, required=True)
    parser.add_argument("--test_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/predictions_annotees")
    return parser.parse_args()


def main():
    args = parse_args()
    model = load_model(args.model_path)
    print(f"Modèle chargé depuis : {args.model_path}")

    _, _, _, _, X_test, y_test_int, class_names, fn_test = load_data_from_dirs(
        args.train_dir, args.val_dir, args.test_dir
    )
    print(f"Classes : {class_names}")

    annotate_and_save_predictions(
        model=model,
        X=X_test,
        y_true=y_test_int,
        filenames=fn_test,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
