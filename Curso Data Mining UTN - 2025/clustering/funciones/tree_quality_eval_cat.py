
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

def tree_quality_eval_cat(y_true, y_pred):
    print("\n=== Evaluación de Árbol de Clasificación ===")
    print("--------------------------------------------")
    
    acc = accuracy_score(y_true, y_pred)
    print(f"Accuracy (exactitud): {acc:.4f}")
    print("Interpretación: Porcentaje de predicciones correctas sobre el total")

    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"\nPrecision (ponderada): {prec:.4f}")
    print("Interpretación: Proporción de aciertos sobre el total de positivos predichos")

    rec = recall_score(y_true, y_pred, average='weighted')
    print(f"\nRecall (sensibilidad, ponderado): {rec:.4f}")
    print("Interpretación: Proporción de verdaderos positivos detectados")

    f1 = f1_score(y_true, y_pred, average='weighted')
    print(f"\nF1 Score (ponderado): {f1:.4f}")
    print("Interpretación: Media armónica entre precisión y recall")

    print("\n=== Matriz de Confusión ===")
    print(confusion_matrix(y_true, y_pred))

    print("\n=== Reporte de Clasificación ===")
    print(classification_report(y_true, y_pred, zero_division=0))