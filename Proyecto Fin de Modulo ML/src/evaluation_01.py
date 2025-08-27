from typing import Dict, Any
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelBinarizer

def evaluar_modelo_resultados(resultados: dict) -> Dict[str, Any]:
    model = resultados["model"]
    X_test = resultados["X_test"]
    y_test = resultados["y_test"]

    y_pred = model.predict(X_test)

    report_txt = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average='macro')

    auc_macro = None
    try:
        y_proba = model.predict_proba(X_test)
        lb = LabelBinarizer().fit(y_test)
        y_test_bin = lb.transform(y_test)
        if y_test_bin.ndim == 1:
            y_test_bin = np.column_stack((1 - y_test_bin, y_test_bin))
        if y_test_bin.shape[1] > 1:
            auc_macro = roc_auc_score(y_test_bin, y_proba, average='macro', multi_class='ovr')
    except Exception:
        pass

    print("\nClassification report:\n", report_txt)
    print("\nMatriz de confusión:\n", cm)
    print("\nMétricas adicionales:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score (macro): {f1m:.4f}")
    if auc_macro is not None:
        print(f"AUC OvR (macro): {auc_macro:.4f}")

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de confusión (Modelo 01)')
    plt.xlabel('Predicho'); plt.ylabel('Real')
    plt.tight_layout(); plt.show()

    return {
        "classification_report": report_txt,
        "confusion_matrix": cm,
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        **({"roc_auc_ovr_macro": float(auc_macro)} if auc_macro is not None else {})
    }