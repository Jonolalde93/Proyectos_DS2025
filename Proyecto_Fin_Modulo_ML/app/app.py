import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Predicción Modelos Complaints", layout="wide")

def _patch_sklearn_privates():
    try:
        import sklearn.utils as _sku
        try:
            from sklearn.compose._column_transformer import _get_column_indices as _gci
            if not hasattr(_sku, "_get_column_indices"):
                setattr(_sku, "_get_column_indices", _gci)
        except Exception:
            pass
        if not hasattr(_sku, "_RemainderColsList"):
            class _RemainderColsList(list):
                pass
            setattr(_sku, "_RemainderColsList", _RemainderColsList)
    except Exception:
        pass

APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
MODELS_DIR = ROOT_DIR / "models"

DEFAULT_PATHS = {
    "Modelo 01": str(MODELS_DIR / "final_model_01.pkl"),
    "Modelo 02": str(MODELS_DIR / "final_model_02.pkl"),
    "Modelo 03": str(MODELS_DIR / "final_model_03.pkl"),
}

LABELS_M1 = ['No', 'Pending', 'Yes']
LABELS_M2 = ['No', 'Yes']
LABELS_M3 = {0: "<= 3 días", 1: "4–14 días", 2: "> 14 días"}

EXCLUDE_COLS = {
    "Modelo 01": ["ZIP code"],
    "Modelo 02": ["ZIP code"],
    "Modelo 03": ["ZIP code"],
}
EXCLUDE_ALIASES = ["ZIP Code", "Zip code", "Zip Code"]

st.sidebar.title("🔧 Configuración")
modelo_seleccion = st.sidebar.selectbox("Elige el modelo", list(DEFAULT_PATHS.keys()))
ruta_modelo = st.sidebar.text_input("Ruta del .pkl", DEFAULT_PATHS[modelo_seleccion])

if modelo_seleccion == "Modelo 02":
    thr = st.sidebar.slider("Threshold (clase 'Yes')", 0.0, 1.0, 0.50, 0.01)
else:
    thr = None

with st.sidebar.expander("📁 Modelos disponibles en /models", expanded=False):
    if MODELS_DIR.exists():
        archivos = sorted([p.name for p in MODELS_DIR.glob("*.pkl")])
        st.write(archivos if archivos else "No hay .pkl en models/")
    else:
        st.write(f"No existe: {MODELS_DIR}")

WHAT_IT_PREDICTS = {
    "Modelo 01": "¿El consumidor disputará la queja?",
    "Modelo 02": "¿La queja sera procesada a tiempo?",
    "Modelo 03": "Clasificación del tiempo de respuesta",
}

st.title("📦 Predicción con modelos entrenados")
st.markdown(f"**Modelo seleccionado:** `{modelo_seleccion}`")
st.subheader(WHAT_IT_PREDICTS[modelo_seleccion])

@st.cache_resource(show_spinner=True)
def load_model(path: str):
    if modelo_seleccion.strip().lower() == "modelo 03":
        _patch_sklearn_privates()
    if not os.path.exists(path):
        raise FileNotFoundError(f"No se encontró el archivo del modelo: {path}")
    model = joblib.load(path)
    expected_cols = None
    preproc = None
    try:
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            preproc = model.named_steps["preprocessor"]
            if hasattr(preproc, "feature_names_in_"):
                expected_cols = list(preproc.feature_names_in_)
    except Exception:
        pass
    return model, preproc, expected_cols

try:
    model, preproc, expected_cols = load_model(ruta_modelo)
except Exception as e:
    st.error(str(e))
    st.stop()

with st.expander("🔎 Columnas esperadas por el preprocesador", expanded=False):
    if expected_cols:
        st.write(expected_cols)
    else:
        st.info("No se detectaron automáticamente las columnas esperadas.")

def get_preproc_info(preproc):
    input_cols, cat_cols, cat_choices, num_cols = None, [], {}, []
    if preproc is None:
        return input_cols, cat_cols, cat_choices, num_cols
    try:
        if hasattr(preproc, "feature_names_in_"):
            input_cols = list(preproc.feature_names_in_)
        transformers = getattr(preproc, "transformers_", [])
        from sklearn.pipeline import Pipeline
        try:
            from sklearn.preprocessing import OneHotEncoder
        except Exception:
            OneHotEncoder = None
        for name, transformer, cols in transformers:
            if name == "cat":
                cat_cols = list(cols)
                ohe = None
                if isinstance(transformer, Pipeline):
                    for _, step in transformer.steps:
                        if OneHotEncoder is not None and isinstance(step, OneHotEncoder):
                            ohe = step; break
                        if hasattr(step, "categories_"):
                            ohe = step; break
                else:
                    if hasattr(transformer, "categories_"):
                        ohe = transformer
                if ohe is not None and hasattr(ohe, "categories_"):
                    for c, cats in zip(cat_cols, ohe.categories_):
                        cat_choices[c] = [str(x) for x in list(cats)]
            if name == "num":
                num_cols = list(cols)
    except Exception:
        pass
    return input_cols, cat_cols, cat_choices, num_cols

input_cols_all, cat_cols_all, cat_choices, num_cols_all = get_preproc_info(preproc)

exclude = set(EXCLUDE_COLS.get(modelo_seleccion, []))
if input_cols_all:
    for alias in EXCLUDE_ALIASES:
        if alias in input_cols_all:
            exclude.add(alias)
    input_cols_ui = [c for c in input_cols_all if c not in exclude]
else:
    input_cols_ui = []

def predict_model_01(df: pd.DataFrame):
    y_num = model.predict(df)
    if getattr(y_num, "dtype", None) and y_num.dtype.kind in ("U", "S", "O"):
        return y_num, None
    mapped = [LABELS_M1[i] if i < len(LABELS_M1) else str(i) for i in y_num]
    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(df)
        except Exception:
            y_proba = None
    return np.array(mapped), y_proba

def predict_model_02(df: pd.DataFrame, threshold: float):
    if hasattr(model, "predict_proba"):
        p_yes = model.predict_proba(df)[:, 1]
        y_hat = (p_yes >= threshold).astype(int)
        labels = np.array([LABELS_M2[i] for i in y_hat])
        return labels, p_yes
    else:
        y_hat = model.predict(df)
        labels = np.array([LABELS_M2[int(i)] for i in y_hat])
        return labels, None

def predict_model_03(df: pd.DataFrame):
    y_num = model.predict(df)
    labels = np.array([LABELS_M3.get(int(i), str(i)) for i in y_num])
    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(df)
        except Exception:
            proba = None
    return labels, proba

def sanitize_for_model(
    df: pd.DataFrame,
    expected_cols: list | None,
    cat_cols_all: list | None,
    excluded: set | None = None,
    num_cols_all: list | None = None
) -> pd.DataFrame:
    import pandas as _pd
    import numpy as _np

    def _is_missing(v):
        try:
            return _pd.isna(v)
        except Exception:
            return False

    excluded = set(excluded or set())
    cat_set = set(cat_cols_all or [])
    num_set = set(num_cols_all or [])

    dfx = df.copy()

    if expected_cols is not None:
        for c in expected_cols:
            if c not in dfx.columns:
                dfx[c] = _np.nan
        for c in excluded:
            if c in expected_cols:
                dfx[c] = _np.nan
        dfx = dfx[expected_cols]

    for c in dfx.columns:
        if c in excluded:
            continue
        if c in cat_set:
            dfx[c] = dfx[c].astype("object")
            dfx[c] = dfx[c].apply(
                lambda v: _np.nan if _is_missing(v) or (isinstance(v, str) and v.strip() == "") else str(v)
            )
        else:
            dfx[c] = _pd.to_numeric(dfx[c], errors="coerce")

    return dfx

st.markdown("---")
tab_manual, tab_batch = st.tabs(["🧪 Predicción manual", "📤 Batch por CSV"])

with tab_manual:
    st.subheader("🧪 Predicción manual")
    if not input_cols_all:
        st.warning("No se pudieron detectar automáticamente las columnas de entrada.")
    else:
        with st.expander("📋 Variables usadas por el modelo", expanded=False):
            st.write(input_cols_ui)
        excl_list = sorted(list(exclude))
        if excl_list:
            with st.expander("🚫 Variables ocultas (se envían como NaN si el modelo las espera)", expanded=False):
                st.write(excl_list)
        with st.form("manual_form"):
            st.caption("Introduce los valores para un único registro. Las columnas no especificadas se dejarán como NaN.")
            user_inputs = {}
            for col in input_cols_ui:
                if col in (cat_cols_all or []):
                    choices = cat_choices.get(col, [])
                    if choices:
                        sel = st.selectbox(col, options=["(vacío)"] + choices, index=0)
                        user_inputs[col] = None if sel == "(vacío)" else sel
                    else:
                        val = st.text_input(f"{col} (texto)")
                        user_inputs[col] = (val if val.strip() != "" else None)
                else:
                    val = st.text_input(f"{col} (numérico)", value="")
                    user_inputs[col] = (val if val.strip() != "" else None)
            submitted = st.form_submit_button("Predecir")
        if submitted:
            row = {}
            for col in input_cols_ui:
                val = user_inputs.get(col, None)
                if col in (cat_cols_all or []):
                    row[col] = np.nan if val is None else val
                else:
                    try:
                        row[col] = float(val) if val is not None else np.nan
                    except Exception:
                        row[col] = np.nan
            df_one = pd.DataFrame([row], columns=input_cols_ui)
            df_one = sanitize_for_model(df_one, expected_cols, cat_cols_all, exclude, num_cols_all)
            try:
                if modelo_seleccion == "Modelo 01":
                    preds, proba = predict_model_01(df_one)
                    st.success(f"Predicción: **{preds[0]}**")
                    if proba is not None:
                        st.write("Probabilidades por clase:")
                        st.write(pd.DataFrame(proba, columns=[f"Proba_{i}" for i in range(proba.shape[1])]))
                elif modelo_seleccion == "Modelo 02":
                    preds, p_yes = predict_model_02(df_one, thr if thr is not None else 0.5)
                    st.success(f"Predicción: **{preds[0]}**")
                    if p_yes is not None:
                        st.write(f"Probabilidad de 'Yes': {float(p_yes[0]):.4f}  (threshold={thr:.2f})")
                else:
                    preds, proba = predict_model_03(df_one)
                    st.success(f"Predicción: **{preds[0]}**")
                    if proba is not None:
                        st.write("Probabilidades por clase:")
                        st.write(pd.DataFrame(proba, columns=[f"Proba_{i}" for i in range(proba.shape[1])]))
            except Exception as e:
                st.error(f"Error durante la predicción manual: {e}")
                with st.expander("Depuración (dtypes del DataFrame enviado al modelo)"):
                    st.write(df_one.dtypes.astype(str))

with tab_batch:
    st.subheader("📤 Sube un CSV para predicción (batch)")
    uploaded = st.file_uploader("CSV con el mismo esquema de columnas usado en entrenamiento.", type=["csv"])
    if uploaded is not None:
        try:
            df_in = pd.read_csv(uploaded)
            st.write("Primeras filas del CSV:")
            st.dataframe(df_in.head())
            df_pred = sanitize_for_model(df_in.copy(), expected_cols, cat_cols_all, exclude, num_cols_all)
            with st.spinner("Calculando predicciones..."):
                if modelo_seleccion == "Modelo 01":
                    preds, proba = predict_model_01(df_pred)
                elif modelo_seleccion == "Modelo 02":
                    preds, proba = predict_model_02(df_pred, thr if thr is not None else 0.5)
                else:
                    preds, proba = predict_model_03(df_pred)
            out = df_in.copy()
            col_name = {
                "Modelo 01": "Pred_ConsumerDisputed",
                "Modelo 02": "Pred_TimelyResponse",
                "Modelo 03": "Pred_DelayClass",
            }[modelo_seleccion]
            out[col_name] = preds
            if proba is not None:
                if modelo_seleccion == "Modelo 02":
                    out["Proba_Yes"] = proba
                else:
                    for i in range(proba.shape[1]):
                        out[f"Proba_{i}"] = proba[:, i]
            st.subheader("✅ Predicciones")
            st.dataframe(out.head(50))
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Descargar predicciones (CSV)",
                data=csv,
                file_name=f"predicciones_{modelo_seleccion.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            st.subheader("📊 Resumen de clases")
            st.write(out[col_name].value_counts(dropna=False))
        except Exception as e:
            st.error(f"Ocurrió un error durante la predicción: {e}")
            with st.expander("Depuración (dtypes del DataFrame enviado al modelo)"):
                try:
                    st.write(df_pred.dtypes.astype(str))
                except Exception:
                    st.info("No se pudo mostrar dtypes de df_pred.")
    else:
        st.info("Sube un CSV para obtener predicciones.")

