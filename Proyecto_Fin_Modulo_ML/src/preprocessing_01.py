from typing import Tuple, List, Dict
import pandas as pd

def get_labels() -> List[str]:
    return ['No', 'Pending', 'Yes']

def get_features_with_subproduct() -> List[str]:
    return [
        'Product', 'Sub-product', 'Issue', 'State',
        'Company_grouped_filtered', 'Company response', 'Timely response?',
        'Difference in days'
    ]

def prepare_data(dt_quejas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int], List[str]]:
    """
    Construye X e y (numéricas) a partir de dt_quejas usando las columnas del notebook.
    Devuelve X, y_num, label_to_num y labels (en ese orden).
    """
    labels = get_labels()
    features_with_subproduct = get_features_with_subproduct()

    X = dt_quejas[features_with_subproduct].copy()
    y = dt_quejas['Consumer disputed?'].copy()

    label_to_num = {label: i for i, label in enumerate(labels)}
    y_num = y.map(label_to_num)

    return X, y_num, label_to_num, labels
