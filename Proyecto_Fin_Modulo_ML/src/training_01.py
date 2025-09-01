from typing import Tuple, List, Optional
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
import joblib


def entrenar_xgb_pesos(
    X: pd.DataFrame,
    y: Optional[pd.Series] = None,
    labels: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: Optional[str] = None
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Entrena el pipeline (OneHotEncoder + XGBClassifier).

    Uso 1 (con datos ya preparados):
        entrenar_xgb_pesos(X, y, labels)

    Uso 2 (modo conveniente con DataFrame crudo del notebook):
        entrenar_xgb_pesos(dt_quejas)
    -> Internamente llama a preprocessing_01.prepare_data.
    """
    if y is None:
        import preprocessing_01  
        X, y, _, labels = preprocessing_01.prepare_data(X)

    if labels is None:
        raise ValueError("Faltan 'labels'. Si pasas X e y manualmente, debes pasar también 'labels'.")

    cat_features = X.select_dtypes(include='object').columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)],
        remainder='passthrough'
    )

    counts = Counter(y_train)
    total = float(sum(counts.values()))
    pesos = {cls: total / count for cls, count in counts.items()}
    sample_weight = y_train.map(pesos).to_numpy()

    xgb = XGBClassifier(
        objective='multi:softprob',
        num_class=len(labels),
        eval_metric='mlogloss',
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=random_state
    )

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', xgb)
    ])

    pipe.fit(X_train, y_train, classifier__sample_weight=sample_weight)

    if model_path is not None:
        joblib.dump(pipe, model_path)

    return pipe, X_train, X_test, y_train, y_test

def entrenar_xgb_pesos_resultados(
    X: pd.DataFrame,
    y: pd.Series,
    labels: List[str],
    test_size: float = 0.2,
    random_state: int = 42,
    model_path: Optional[str] = None
) -> dict:
    """
    Wrapper que devuelve un diccionario 'resultados' como en el Modelo 02.
    No toca la función original 'entrenar_xgb_pesos'.
    """
    model, X_train, X_test, y_train, y_test = entrenar_xgb_pesos(
        X=X, y=y, labels=labels,
        test_size=test_size, random_state=random_state, model_path=model_path
    )
    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "labels": labels
    }