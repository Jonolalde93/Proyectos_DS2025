from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

def evaluar_xgb_delay(resultados: Dict[str, Any]) -> Dict[str, Any]:
    best_model = resultados["best_model"]
    rs = resultados["search"]
    X_test = resultados["X_test"]
    y_test = resultados["y_test"]
    y_pred = resultados["y_pred"]

    print("Mejores parámetros:", rs.best_params_)
    print("\nClassification report:\n", classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatriz de confusión:\n", cm)
    kappa = cohen_kappa_score(y_test, y_pred, weights="quadratic")
    print("Quadratic Weighted Kappa:", kappa)

    auc_value = None
    if hasattr(best_model.named_steps["clf"], "predict_proba"):
        y_proba = best_model.predict_proba(X_test)
        classes_ = best_model.named_steps["clf"].classes_
        if len(classes_) == 2:
            auc_value = roc_auc_score(y_test, y_proba[:, 1])
            print(f"ROC AUC (binario): {auc_value:.4f}")
        else:
            y_test_bin = label_binarize(y_test, classes=classes_)
            auc_value = roc_auc_score(y_test_bin, y_proba, average="macro", multi_class="ovr")
            print(f"ROC AUC (multiclase, OVR-macro): {auc_value:.4f}")

    return {
        "best_params": dict(rs.best_params_),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": cm,
        "quadratic_weighted_kappa": float(kappa),
        "roc_auc": (float(auc_value) if auc_value is not None else None)
    }