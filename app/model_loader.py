import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model


@dataclass
class LoadedArtifacts:
    model_file: str
    keras_model: Any
    expected_cols: List[str]
    target_names: List[str]


def _load_expected_cols_from_train_csv() -> List[str]:
    # Coincide con la referencia: si existe X_train_final.csv, usar exactamente su orden de columnas.
    train_cols_path = os.getenv("TRAIN_COLS_PATH", "datasets_finales/X_train_final.csv")
    path = Path(train_cols_path)

    if not path.exists():
        return []

    header_cols = pd.read_csv(path, nrows=0).columns.tolist()
    return [str(col) for col in header_cols]


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

    if not feature_cols:
        raise ValueError("El PKL no contiene 'feature_cols'.")

    if not target_names:
        raise ValueError("El PKL no contiene 'target_names'.")

    expected_cols = _load_expected_cols_from_train_csv() or [str(col) for col in feature_cols]

    return LoadedArtifacts(
        model_file=str(keras_path.name),
        keras_model=keras_model,
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
