from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight


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

    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights = y_train.map(class_weight_dict)

    xgb = XGBClassifier(
        objective="multi:softprob",
        num_class=len(classes),
        eval_metric="mlogloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=random_state
    )

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("clf", xgb)
    ])

    param_grid = {
        "clf__n_estimators": [200, 500],
        "clf__max_depth": [5, 10, 20],
        "clf__learning_rate": [0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1_macro",       
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )

    grid.fit(X_train, y_train, clf__sample_weight=sample_weights)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    resultados = {
        "best_model": best_model,
        "search": grid,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "class_weight_dict": class_weight_dict
    }
    return resultados