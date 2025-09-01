from typing import Dict, Any
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def evaluar_xgb_weight_timely(resultados: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calcula y muestra las mismas métricas/visualizaciones que en el notebook
    usando los objetos devueltos por entrenar_xgb_weight_timely.
    """
    y_test = resultados['y_test']
    y_pred = resultados['y_pred']
    y_proba_yes = resultados['y_proba_yes']
    threshold = resultados['threshold']
    grid = resultados['grid']

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print("Matriz de confusión:\n", cm)
    print("\nReporte de clasificación:\n",
          classification_report(y_test, y_pred, target_names=['No', 'Yes']))

    print("\nMétricas adicionales:")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1m = f1_score(y_test, y_pred, average='macro')
    f1w = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_proba_yes)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1-score (macro): {f1m:.4f}")
    print(f"F1-score (weighted): {f1w:.4f}")
    print(f"ROC AUC (binario): {auc:.4f}")

    print(f"\n Mejores parámetros: {grid.best_params_}")
    print(f" Mejor F1-macro (CV): {grid.best_score_:.4f}")

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
    plt.title(f'Matriz de confusión (XGB + scale_pos_weight, thr={threshold})')
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.tight_layout()
    plt.show()

    return {
        "confusion_matrix": cm,
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
        "roc_auc": float(auc),
        "best_params": dict(grid.best_params_),
        "best_cv_f1_macro": float(grid.best_score_)
    }