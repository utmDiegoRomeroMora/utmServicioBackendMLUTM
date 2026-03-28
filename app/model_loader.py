import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


@dataclass
class LoadedArtifacts:
    model_file: str
    keras_model: Any
    scaler: object
    feature_cols: List[str]
    target_names: List[str]


def _resolve_keras_path(model_dir: Path, pkl_data: Dict, default_keras_name: str) -> Path:
    # Prioridad: nombre explicito en entorno, ruta guardada en pkl, fallback local.
    explicit_path = model_dir / default_keras_name
    if explicit_path.exists():
        return explicit_path

    saved_path = pkl_data.get("model_weights_path")
    if saved_path:
        saved = Path(saved_path)
        if saved.exists():
            return saved

        candidate = model_dir / saved.name
        if candidate.exists():
            return candidate

    fallback = model_dir / "modelo_entrenado.keras"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(
        "No se encontro el archivo del modelo Keras (.keras o ruta valida en model_weights_path)."
    )


def load_artifacts(model_dir: Path, pkl_name: str, keras_name: str) -> LoadedArtifacts:
    pkl_path = model_dir / pkl_name
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"No se encontro el archivo PKL en: {pkl_path}. "
            "Copia modelo_entrenado.pkl dentro de la carpeta models/."
        )

    with pkl_path.open("rb") as f:
        pkl_data = pickle.load(f)

    keras_path = _resolve_keras_path(model_dir=model_dir, pkl_data=pkl_data, default_keras_name=keras_name)
    keras_model = load_model(keras_path)

    feature_cols = pkl_data.get("feature_cols", [])
    target_names = pkl_data.get("target_names", [])
    scaler = pkl_data.get("scaler")

    if not feature_cols:
        raise ValueError("El PKL no contiene 'feature_cols'.")

    if not target_names:
        raise ValueError("El PKL no contiene 'target_names'.")

    return LoadedArtifacts(
        model_file=str(keras_path.name),
        keras_model=keras_model,
        scaler=scaler,
        feature_cols=feature_cols,
        target_names=target_names,
    )


def transform_features(input_features: Dict[str, float], artifacts: LoadedArtifacts) -> Tuple[np.ndarray, List[str]]:
    missing = [col for col in artifacts.feature_cols if col not in input_features]

    row = {col: float(input_features.get(col, 0.0)) for col in artifacts.feature_cols}
    df = pd.DataFrame([row], columns=artifacts.feature_cols)

    values = df.values
    if artifacts.scaler is not None:
        values = artifacts.scaler.transform(df)

    return np.asarray(values, dtype=np.float32), missing
