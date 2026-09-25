"""Carga del dataset UCI 544 (Estimation of Obesity Levels).

La primera vez descarga con `ucimlrepo` y guarda una copia en data/obesity_raw.csv;
después lee el CSV local para no depender de internet.
"""
import pandas as pd

from config import DATA_DIR

UCI_ID = 544
RAW_CSV = DATA_DIR / "obesity_raw.csv"
CLF_CSV = DATA_DIR / "clf.csv"   # generado por la etapa 01 (features + NObeyesdad)
REG_CSV = DATA_DIR / "reg.csv"   # generado por la etapa 01 (features sin NObeyesdad + Weight)
TARGET_CLF = "NObeyesdad"
TARGET_REG = "Weight"


def load_raw(force_download: bool = False) -> pd.DataFrame:
    """Devuelve el dataset completo (features + NObeyesdad) como DataFrame."""
    if RAW_CSV.exists() and not force_download:
        return pd.read_csv(RAW_CSV)

    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=UCI_ID)
    df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(RAW_CSV, index=False)
    return df
