from typing import Tuple, Dict
import pandas as pd

def prepare_data_timely(dt_quejas_corr: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int]]:
    """
    Prepara X e y para el modelo 'Timely response?' a partir de dt_quejas_corr,
    aplicando las mismas transformaciones que en el notebook.
    """
    y_str = dt_quejas_corr['Timely response?'].copy()

    X = dt_quejas_corr.drop(columns=[
        'Timely response?', 'Complaint ID', 'Date received', 'Date sent to company',
        'ZIP code', 'Company'
    ]).copy()

    if 'Difference in days' in X.columns:
        X['Difference in days'] = pd.to_numeric(X['Difference in days'], errors='coerce').fillna(0)

    map_y = {'No': 0, 'Yes': 1}
    y = y_str.map(map_y).astype(int)

    return X, y, map_y