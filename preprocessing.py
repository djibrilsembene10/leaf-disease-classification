# preprocessing.py
"""
Chargement et prétraitement des images pour la classification de maladies
foliaires. Les images sont lues depuis trois répertoires (train/val/test),
organisés au format ImageFolder (un sous-dossier par classe), redimensionnées
et normalisées, puis encodées en one-hot pour l'entraînement.
"""

import os
import cv2
import numpy as np
from tensorflow.keras.utils import to_categorical

def load_data_categorical(train_dir, val_dir, test_dir, img_size=(128,128)):
    """
    Charge les images depuis les répertoires train/val/test
    et renvoie X, y et classes.
    """
    class_names = sorted([d for d in os.listdir(train_dir)
                          if not d.startswith('.') and os.path.isdir(os.path.join(train_dir, d))])
    class_dict = {name: idx for idx, name in enumerate(class_names)}

    def _collect_from(dir_path):
        imgs, labels = [], []
        for cname in class_names:
            folder = os.path.join(dir_path, cname)
            for fname in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, fname))
                if img is None:
                    continue
                img = cv2.resize(img, img_size)
                imgs.append(img)
                labels.append(class_dict[cname])
        X = np.array(imgs, dtype=np.float32)/255.0
        y_int = np.array(labels, dtype=np.int32)
        y = to_categorical(y_int, num_classes=len(class_names))
        return X, y, y_int

    X_train, y_train, y_train_int = _collect_from(train_dir)
    X_val, y_val, y_val_int = _collect_from(val_dir)
    X_test, y_test, y_test_int = _collect_from(test_dir)
    return X_train, y_train, y_train_int, X_val, y_val, y_val_int, X_test, y_test, y_test_int, class_names
