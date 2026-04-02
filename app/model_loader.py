import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb


@dataclass
class LoadedArtifacts:
    model_file: str
    model: Any
    expected_cols: List[str]
    target_names: List[str]


def _load_expected_cols_from_train_csv() -> List[str]:
    train_cols_path = os.getenv("TRAIN_COLS_PATH", "datasets_finales/X_train_final.csv")
    path = Path(train_cols_path)

    if not path.exists():
        return []

    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    return [str(col) for col in header_cols]


def _load_target_names() -> List[str]:
    raw = os.getenv("TARGET_NAMES", "Sin sequía,D0,D1,D2,D3,D4")
    return [name.strip() for name in raw.split(",") if name.strip()]


def load_artifacts(model_dir: Path, xgb_name: str) -> LoadedArtifacts:
    xgb_path = model_dir / xgb_name
    if not xgb_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo XGBoost en: {xgb_path}. "
            f"Copia {xgb_name} dentro de la carpeta {model_dir}/."
        )

    model = xgb.XGBClassifier()
    model.load_model(str(xgb_path))

    expected_cols = _load_expected_cols_from_train_csv()
    if not expected_cols:
        raise ValueError(
            "No se pudieron cargar las columnas esperadas desde el CSV de entrenamiento. "
            "Verifica la variable TRAIN_COLS_PATH."
        )

    target_names = _load_target_names()

    return LoadedArtifacts(
        model_file=str(xgb_path.name),
        model=model,
        expected_cols=expected_cols,
        target_names=target_names,
    )


def transform_features(input_features: Dict[str, float], artifacts: LoadedArtifacts) -> np.ndarray:
    missing = [col for col in artifacts.expected_cols if col not in input_features]
    if missing:
        raise ValueError(f"Faltan columnas requeridas para inferencia: {missing}")

    df = pd.DataFrame([input_features])
    df = df[artifacts.expected_cols]
    return np.asarray(df.values)


def transform_batch_features(observations: List[Dict[str, float]], artifacts: LoadedArtifacts) -> np.ndarray:
    for i, obs in enumerate(observations):
        missing = [col for col in artifacts.expected_cols if col not in obs]
        if missing:
            raise ValueError(f"Observacion {i}: faltan columnas requeridas: {missing}")

    df = pd.DataFrame(observations)
    df = df[artifacts.expected_cols]
    return np.asarray(df.values)


def transform_batch_features(
    observations: List[Dict[str, float]],
    artifacts: LoadedArtifacts,
) -> np.ndarray:
    for idx, obs in enumerate(observations):
        missing = [col for col in artifacts.expected_cols if col not in obs]
        if missing:
            raise ValueError(
                f"Faltan columnas requeridas para inferencia en observations[{idx}]: {missing}"
            )

    df = pd.DataFrame(observations)
    df = df[artifacts.expected_cols]
    return np.asarray(df.values)
