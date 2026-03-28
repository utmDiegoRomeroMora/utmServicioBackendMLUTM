import os
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException

from app.model_loader import LoadedArtifacts, load_artifacts, transform_features
from app.schemas import BatchPredictionRequest, MetadataResponse, PredictionRequest, PredictionResponse


MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_PKL = os.getenv("MODEL_PKL", "modelo_entrenado.pkl")
MODEL_KERAS = os.getenv("MODEL_KERAS", "modelo_entrenado.keras")

app = FastAPI(
    title="UTM Sequia API",
    version="1.0.0",
    description="API de inferencia para clasificacion de sequia con modelo Keras + artefactos PKL",
)


CLASS_CATALOG = {
    "Sin sequía": {
        "severity_level": "normal",
        "severity_description": "Condiciones hidricas dentro de parametros normales.",
        "recommendation": "Mantener monitoreo preventivo y uso eficiente del agua.",
    },
    "D0": {
        "severity_level": "anormalmente seco",
        "severity_description": "Inicio de estres hidrico; posible transicion a sequia.",
        "recommendation": "Activar vigilancia temprana y revisar consumo en sectores sensibles.",
    },
    "D1": {
        "severity_level": "sequia moderada",
        "severity_description": "Deficit de humedad con afectacion inicial en cultivos y abasto.",
        "recommendation": "Aplicar medidas de ahorro y priorizar usos esenciales del recurso.",
    },
    "D2": {
        "severity_level": "sequia severa",
        "severity_description": "Deficit importante con impactos claros en agricultura y disponibilidad.",
        "recommendation": "Escalar protocolos de contingencia hidrica y abastecimiento local.",
    },
    "D3": {
        "severity_level": "sequia extrema",
        "severity_description": "Afectacion amplia en sistemas hidricos, productivos y sociales.",
        "recommendation": "Implementar restricciones operativas y acciones urgentes de mitigacion.",
    },
    "D4": {
        "severity_level": "sequia excepcional",
        "severity_description": "Escenario critico con impactos severos y sostenidos.",
        "recommendation": "Declarar respuesta maxima y coordinar apoyo interinstitucional.",
    },
}


def _interpret_class(class_name: str) -> dict:
    default = {
        "severity_level": "indeterminado",
        "severity_description": "No se encontro descripcion para la clase predicha.",
        "recommendation": "Verificar configuracion de target_names en el artefacto PKL.",
    }
    return CLASS_CATALOG.get(class_name, default)


def _build_prediction_response(
    probs: np.ndarray,
    artifacts: LoadedArtifacts,
    return_proba: bool,
    missing_cols_count: int = 0,
) -> PredictionResponse:
    pred_idx = int(np.argmax(probs))
    pred_name = artifacts.target_names[pred_idx]
    confidence = float(probs[pred_idx])
    class_info = _interpret_class(pred_name)

    probabilities = None
    if return_proba:
        probabilities = {
            artifacts.target_names[i]: float(probs[i])
            for i in range(min(len(artifacts.target_names), len(probs)))
        }
        if missing_cols_count > 0:
            probabilities["_warning_missing_cols_filled_with_0"] = float(missing_cols_count)

    return PredictionResponse(
        predicted_class=pred_name,
        predicted_index=pred_idx,
        confidence=confidence,
        severity_level=class_info["severity_level"],
        severity_description=class_info["severity_description"],
        recommendation=class_info["recommendation"],
        possible_mlp_results=artifacts.target_names,
        probabilities=probabilities,
    )


@app.on_event("startup")
def startup_event() -> None:
    artifacts = load_artifacts(model_dir=MODEL_DIR, pkl_name=MODEL_PKL, keras_name=MODEL_KERAS)
    app.state.artifacts = artifacts


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    artifacts: LoadedArtifacts = app.state.artifacts

    available_catalog = {
        name: CLASS_CATALOG.get(
            name,
            {
                "severity_level": "indeterminado",
                "severity_description": "Sin descripcion definida.",
                "recommendation": "Sin recomendacion definida.",
            },
        )
        for name in artifacts.target_names
    }

    return MetadataResponse(
        model_file=artifacts.model_file,
        feature_cols=artifacts.feature_cols,
        target_names=artifacts.target_names,
        objective="Predecir la clase de severidad de sequia: Sin sequia, D0, D1, D2, D3, D4.",
        class_catalog=available_catalog,
        has_scaler=artifacts.scaler is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    artifacts: LoadedArtifacts = app.state.artifacts

    try:
        x, missing_cols = transform_features(payload.features, artifacts)
        probs = artifacts.keras_model.predict(x, verbose=0)
        probs = np.asarray(probs)[0]
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error al transformar o predecir: {exc}") from exc

    return _build_prediction_response(
        probs=probs,
        artifacts=artifacts,
        return_proba=payload.return_proba,
        missing_cols_count=len(missing_cols),
    )


@app.post("/predict_batch", response_model=List[PredictionResponse])
def predict_batch(payload: BatchPredictionRequest) -> List[PredictionResponse]:
    artifacts: LoadedArtifacts = app.state.artifacts

    if not payload.observations:
        raise HTTPException(status_code=400, detail="'observations' no puede estar vacio.")

    try:
        # Preserva el orden de entrada y alinea columnas contra feature_cols.
        x_df = pd.DataFrame(payload.observations)
        x_df = x_df.reindex(columns=artifacts.feature_cols, fill_value=0.0)
        x_df = x_df.astype(float)

        missing_counts = [
            sum(1 for col in artifacts.feature_cols if col not in obs)
            for obs in payload.observations
        ]

        x = x_df.values
        if artifacts.scaler is not None:
            x = artifacts.scaler.transform(x_df)

        all_probs = artifacts.keras_model.predict(x, verbose=0)
        all_probs = np.asarray(all_probs)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Error en procesamiento por lotes: {exc}") from exc

    results: List[PredictionResponse] = []
    for idx, probs in enumerate(all_probs):
        results.append(
            _build_prediction_response(
                probs=probs,
                artifacts=artifacts,
                return_proba=payload.return_proba,
                missing_cols_count=missing_counts[idx],
            )
        )

    return results
