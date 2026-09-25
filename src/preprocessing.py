"""Preprocesamiento base compartido por clasificación, regresión y clustering.

Encoding:
- Binarias (yes/no, Male/Female)      -> 0/1
- Ordinales CAEC y CALC               -> 0..3 (no < Sometimes < Frequently < Always)
- Nominal MTRANS                      -> one-hot (5 columnas; drop_first=False para
                                         que los árboles vean cada categoría)
- Numéricas                           -> se dejan tal cual. El ESCALADO (que KNN y
                                         K-Means sí necesitan) se hace dentro del
                                         Pipeline de cada modelo, ajustado sólo con
                                         el train, para no filtrar información del test.

Se eliminan las filas duplicadas exactas (24) antes de cualquier split.
"""
import pandas as pd

from src.data import TARGET_CLF, TARGET_REG, CLF_CSV, REG_CSV, load_raw

# Orden "natural" de las clases (de menor a mayor IMC). Se usa en todos los gráficos.
CLASS_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

BINARY_MAP = {
    "Gender": {"Female": 0, "Male": 1},
    "family_history_with_overweight": {"no": 0, "yes": 1},
    "FAVC": {"no": 0, "yes": 1},
    "SMOKE": {"no": 0, "yes": 1},
    "SCC": {"no": 0, "yes": 1},
}
ORDINAL_MAP = {"no": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
ORDINAL_COLS = ["CAEC", "CALC"]
NOMINAL_COLS = ["MTRANS"]
NUMERIC_COLS = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Codifica todas las columnas categóricas. No toca NObeyesdad."""
    out = df.copy()
    for col, mapping in BINARY_MAP.items():
        out[col] = out[col].map(mapping).astype("int8")
    for col in ORDINAL_COLS:
        out[col] = out[col].map(ORDINAL_MAP).astype("int8")
    out = pd.get_dummies(out, columns=NOMINAL_COLS, dtype="int8")
    return out


def build_datasets(df_raw: pd.DataFrame):
    """Devuelve (df_clf, df_reg) listos para modelar.

    df_clf: todas las features codificadas + NObeyesdad (target).
    df_reg: todas las features codificadas SIN NObeyesdad, con Weight como target.
    """
    df = df_raw.drop_duplicates().reset_index(drop=True)
    encoded = encode_features(df)

    df_clf = encoded.copy()
    df_clf[TARGET_CLF] = pd.Categorical(df_clf[TARGET_CLF], categories=CLASS_ORDER)

    df_reg = encoded.drop(columns=[TARGET_CLF])
    # Weight al final para dejar claro que es el target
    df_reg = df_reg[[c for c in df_reg.columns if c != TARGET_REG] + [TARGET_REG]]
    return df_clf, df_reg


def cargar_clf() -> pd.DataFrame:
    """Dataset de clasificación: lee data/clf.csv o lo regenera si no existe."""
    df = pd.read_csv(CLF_CSV) if CLF_CSV.exists() else build_datasets(load_raw())[0]
    df[TARGET_CLF] = pd.Categorical(df[TARGET_CLF], categories=CLASS_ORDER)
    return df


def cargar_reg() -> pd.DataFrame:
    """Dataset de regresión: lee data/reg.csv o lo regenera si no existe."""
    df = pd.read_csv(REG_CSV) if REG_CSV.exists() else build_datasets(load_raw())[1]
    assert TARGET_CLF not in df.columns, "NObeyesdad no debe estar en el dataset de regresión"
    return df
