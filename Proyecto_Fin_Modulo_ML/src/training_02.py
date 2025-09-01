from typing import Dict, Any
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

def entrenar_xgb_weight_timely(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Replica el flujo del notebook:
      - split estratificado
      - cálculo de scale_pos_weight
      - pipeline (OHE + XGBClassifier)
      - GridSearchCV con scoring f1_macro
      - predicciones con umbral 'threshold' sobre test
    Devuelve un diccionario con el mejor modelo, grid y resultados de test.
    """
    cat_cols = X.select_dtypes(include='object').columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=random_state
    )

    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    spw = (neg / pos) if pos > 0 else 1.0

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )

    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            random_state=random_state,
            n_jobs=-1,
            scale_pos_weight=spw
        ))
    ])

    param_grid = {
        'clf__n_estimators': [300, 500],
        'clf__max_depth': [4, 6],
        'clf__learning_rate': [0.05, 0.1],
        'clf__subsample': [0.8, 1.0],
        'clf__colsample_bytree': [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring='f1_macro',
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_proba_yes = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba_yes >= threshold).astype(int)

    resultados = {
        'best_model': best_model,
        'grid': grid,
        'threshold': threshold,
        'X_test': X_test,
        'y_test': y_test,
        'y_proba_yes': y_proba_yes,
        'y_pred': y_pred,
        'scale_pos_weight': spw,
        'pos_count_train': pos,
        'neg_count_train': neg
    }
    return resultados
