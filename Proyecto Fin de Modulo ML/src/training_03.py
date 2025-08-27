from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def entrenar_xgb_delay(
    X: pd.DataFrame,
    y: pd.Series,
    random_state: int = 42
) -> Dict[str, Any]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=random_state
    )

    cat_cols = X_train.select_dtypes(include="object").columns.tolist()
    num_cols = X_train.select_dtypes(exclude="object").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols)
        ]
    )

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(np.unique(y)),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=42
    )

    pipe = ImbPipeline(steps=[
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("clf", xgb)
    ])

    param_dist = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [5, 10, 20],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    rs = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=48,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=42,
        refit=True
    )

    rs.fit(X_train, y_train)
    best_model = rs.best_estimator_
    y_pred = best_model.predict(X_test)

    resultados = {
        "best_model": best_model,
        "search": rs,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred
    }
    return resultados