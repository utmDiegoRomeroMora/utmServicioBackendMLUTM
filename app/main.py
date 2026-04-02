import os
from pathlib import Path
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.model_loader import LoadedArtifacts, load_artifacts, transform_batch_features, transform_features
from app.schemas import BatchPredictionRequest, MetadataResponse, PredictionRequest, PredictionResponse


MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MODEL_XGB = os.getenv("MODEL_XGB", "xgb_t1_estricta_ex_ante_sinmun.json")

app = FastAPI(
    title="UTM Sequia API",
    version="1.0.0",
    description="API de inferencia para clasificacion de sequia con modelo XGBoost",
)

cors_origins_raw = os.getenv("CORS_ALLOW_ORIGINS", "https://utm-front-end-diego-romero.vercel.app")
if cors_origins_raw.strip() == "*":
    cors_origins = ["*"]
    allow_credentials = False
else:
    cors_origins = [origin.strip() for origin in cors_origins_raw.split(",") if origin.strip()]
    allow_credentials = True

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
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
    """Interpreta una clase predicha y devuelve su catalogo descriptivo.

  
    1. Recibe el nombre de la clase predicha por el modelo.
    2. Define un bloque `default` para clases no registradas.
    3. Busca la clase en `CLASS_CATALOG`.
    4. Retorna la informacion encontrada o el bloque `default`.
    """
    # Interpreta la clase predicha y devuelve metadatos de severidad y recomendacion.
    print(f"[INFO] _interpret_class: interpretando clase '{class_name}'")
    default = {
        "severity_level": "indeterminado",
        "severity_description": "No se encontro descripcion para la clase predicha.",
        "recommendation": "Verificar configuracion de target_names en la variable de entorno.",
    }
    return CLASS_CATALOG.get(class_name, default)


def _build_prediction_response(
    probs: np.ndarray,
    artifacts: LoadedArtifacts,
    return_proba: bool,
) -> PredictionResponse:
    """Construye la respuesta final de prediccion para una observacion.

  
    1. Calcula el indice con mayor probabilidad usando `argmax`.
    2. Obtiene el nombre de la clase y su confianza.
    3. Enriquese la salida con severidad y recomendacion mediante `_interpret_class`.
    4. Si `return_proba` es True, arma el diccionario de probabilidades por clase.
    5. Retorna un objeto `PredictionResponse` con toda la informacion.
    """
    # Construye la respuesta final de prediccion para una observacion.
    print(
        "[INFO] _build_prediction_response: construyendo respuesta "
        f"(return_proba={return_proba}, total_clases={len(artifacts.target_names)})"
    )
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
    """Inicializa los artefactos del modelo cuando arranca la API.

  
    1. Lee la configuracion de rutas y nombres de artefactos.
    2. Carga modelo Keras y artefactos auxiliares con `load_artifacts`.
    3. Guarda los artefactos en `app.state.artifacts` para reutilizarlos.
    4. Registra en consola un resumen de la carga.
    """
    # Carga modelo y artefactos una sola vez al iniciar el servidor.
    print(
        "[INFO] startup_event: iniciando carga de artefactos "
        f"(MODEL_DIR='{MODEL_DIR}', MODEL_XGB='{MODEL_XGB}')"
    )
    artifacts = load_artifacts(model_dir=MODEL_DIR, xgb_name=MODEL_XGB)
    app.state.artifacts = artifacts
    print(
        "[INFO] startup_event: artefactos cargados correctamente "
        f"(model_file='{artifacts.model_file}', total_features={len(artifacts.expected_cols)}, "
        f"total_clases={len(artifacts.target_names)})"
    )


@app.get("/health")
def health() -> dict:
    """Verifica disponibilidad basica del servicio.

  
    1. Registra en consola que se consulto el estado.
    2. Retorna un objeto simple con `status: ok`.
    """
    # Endpoint de verificacion basica para saber si la API esta viva.
    print("[INFO] /health: verificacion de estado solicitada")
    return {"status": "ok"}


@app.get("/metadata", response_model=MetadataResponse)
def metadata() -> MetadataResponse:
    """Expone metadatos del modelo y el catalogo de clases disponibles.

  
    1. Obtiene los artefactos cargados desde `app.state.artifacts`.
    2. Construye un catalogo de clases solo para `target_names` disponibles.
    3. Completa datos de respaldo para clases sin descripcion definida.
    4. Retorna un `MetadataResponse` con columnas, clases y objetivo del modelo.
    """
    # Endpoint que expone metadatos del modelo, columnas esperadas y clases objetivo.
    print("[INFO] /metadata: solicitud de metadatos del modelo")
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

    print(
        "[INFO] /metadata: metadatos preparados "
        f"(features={len(artifacts.expected_cols)}, clases={len(artifacts.target_names)})"
    )

    return MetadataResponse(
        model_file=artifacts.model_file,
        feature_cols=artifacts.expected_cols,
        target_names=artifacts.target_names,
        objective="Predecir la clase de severidad de sequia: Sin sequia, D0, D1, D2, D3, D4.",
        class_catalog=available_catalog,
        has_scaler=False,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    """Realiza inferencia para una sola observacion.

  
    1. Recibe el payload con features y bandera `return_proba`.
    2. Transforma las features al formato esperado por el modelo.
    3. Ejecuta la prediccion del modelo Keras.
    4. Convierte la salida a `numpy` y toma el vector de probabilidades.
    5. Construye y retorna la respuesta con `_build_prediction_response`.
    6. Si ocurre un error, responde con HTTP 400 y detalle del problema.
    """
    # Endpoint para inferencia individual con una sola observacion.
    print(
        "[INFO] /predict: solicitud recibida "
        f"(features_recibidas={len(payload.features)}, return_proba={payload.return_proba})"
    )
    artifacts: LoadedArtifacts = app.state.artifacts

    try:
        # Transforma las features al formato esperado por el modelo.
        x = transform_features(payload.features, artifacts)
        print(f"[INFO] /predict: transformacion completada con shape={x.shape}")

        # Ejecuta la prediccion para una sola observacion.
        probs = artifacts.model.predict_proba(x)
        probs = np.asarray(probs)[0]
        print("[INFO] /predict: prediccion completada")
    except Exception as exc:
        print(f"[ERROR] /predict: fallo en transformacion/prediccion -> {exc}")
        raise HTTPException(status_code=400, detail=f"Error al transformar o predecir: {exc}") from exc

    return _build_prediction_response(
        probs=probs,
        artifacts=artifacts,
        return_proba=payload.return_proba,
    )


@app.post("/predict_batch", response_model=List[PredictionResponse])
def predict_batch(payload: BatchPredictionRequest) -> List[PredictionResponse]:
    """Realiza inferencia para multiples observaciones en un solo request.

  
    1. Recibe el lote de observaciones y valida que no este vacio.
    2. Transforma todas las observaciones con `transform_batch_features`.
    3. Ejecuta prediccion en lote con el modelo Keras.
    4. Recorre cada vector de probabilidades del resultado.
    5. Construye una `PredictionResponse` por observacion.
    6. Retorna la lista final de respuestas.
    7. Si ocurre un error, responde con HTTP 400 y detalle del problema.
    """
    # Endpoint para inferencia por lotes con multiples observaciones.
    print(
        "[INFO] /predict_batch: solicitud recibida "
        f"(observaciones={len(payload.observations)}, return_proba={payload.return_proba})"
    )
    artifacts: LoadedArtifacts = app.state.artifacts

    if not payload.observations:
        print("[WARN] /predict_batch: se recibio una lista vacia de observaciones")
        raise HTTPException(status_code=400, detail="'observations' no puede estar vacio.")

    try:
        # Transforma todo el lote al formato esperado por el modelo.
        x = transform_batch_features(payload.observations, artifacts)
        print(f"[INFO] /predict_batch: transformacion por lotes completada con shape={x.shape}")

        # Ejecuta inferencia para todo el lote en una sola llamada.
        all_probs = artifacts.model.predict_proba(x)
        all_probs = np.asarray(all_probs)
        print(f"[INFO] /predict_batch: prediccion por lotes completada (filas={len(all_probs)})")
    except Exception as exc:
        print(f"[ERROR] /predict_batch: fallo en procesamiento por lotes -> {exc}")
        raise HTTPException(status_code=400, detail=f"Error en procesamiento por lotes: {exc}") from exc

    # Construye una respuesta por cada observacion del lote.
    results: List[PredictionResponse] = []
    for idx, probs in enumerate(all_probs):
        print(f"[INFO] /predict_batch: construyendo respuesta para indice={idx}")
        results.append(
            _build_prediction_response(
                probs=probs,
                artifacts=artifacts,
                return_proba=payload.return_proba,
            )
        )

    print(f"[INFO] /predict_batch: total_respuestas={len(results)}")
    return results
