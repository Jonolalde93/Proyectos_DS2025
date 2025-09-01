import pandas as pd
from typing import Tuple

def categorizar_dias(x):
    if x <= 3:
        return 0
    elif x <= 14:
        return 1
    else:
        return 2

def prepare_data_delay(dt_quejas: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    dt_quejas = dt_quejas.copy()
    dt_quejas["Delay_Class"] = dt_quejas["Difference in days"].apply(categorizar_dias)

    cols_quitar = [
        'Difference in days','Complaint ID','Date received','Date sent to company',
        'Company','ZIP code','Delay_Class'
    ]
    X = dt_quejas.drop(columns=cols_quitar, errors='ignore')
    y = dt_quejas['Delay_Class']

    return X, y