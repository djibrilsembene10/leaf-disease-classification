# evaluate.py
"""
Fonctions d'évaluation et de visualisation : courbes d'apprentissage,
matrice de confusion, rapport de classification (précision/recall/F1) et
courbes ROC multi-classes.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, roc_curve, auc

def plot_history(history, best_epoch):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Perte')
    plt.xlabel('Époque')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.axvline(best_epoch, color='r', linestyle='--', label=f'Best Epoch ({best_epoch})')
    plt.title('Précision')
    plt.xlabel('Époque')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, y_test_int, class_names, pred_probs=None):
    if pred_probs is None:
        pred_probs = model.predict(X_test)
    pred_classes = np.argmax(pred_probs, axis=1)

    test_acc = accuracy_score(y_test_int, pred_classes)
    test_f1 = f1_score(y_test_int, pred_classes, average='macro')

    print(f"\n🎯 Test Accuracy: {test_acc:.4f}")
    print(f"🎯 Test F1-score: {test_f1:.4f}")

    cm = confusion_matrix(y_test_int, pred_classes)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.title('Matrice de confusion')
    plt.show()

    print("\nClassification Report:")
    print(classification_report(y_test_int, pred_classes, target_names=class_names))

    plt.figure(figsize=(6,5))
    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_test[:,i], pred_probs[:,i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_names[i]} (AUC={roc_auc:.4f})')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.title("ROC multi-classes")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()
