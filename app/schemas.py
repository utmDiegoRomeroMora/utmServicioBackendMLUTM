from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    features: Dict[str, float] = Field(
        ..., description="Diccionario con columnas y valores numericos del modelo"
    )
    return_proba: bool = Field(
        default=True, description="Si es true, devuelve probabilidades por clase"
    )


class BatchPredictionRequest(BaseModel):
    observations: List[Dict[str, float]] = Field(
        ..., description="Lista de diccionarios con las columnas del CSV"
    )
    return_proba: bool = Field(
        default=True, description="Si es true, devuelve probabilidades para cada fila"
    )


class PredictionResponse(BaseModel):
    predicted_class: str
    predicted_index: int
    confidence: float
    severity_level: str
    severity_description: str
    recommendation: str
    possible_mlp_results: List[str]
    probabilities: Optional[Dict[str, float]] = None


class MetadataResponse(BaseModel):
    model_file: str
    feature_cols: List[str]
    target_names: List[str]
    objective: str
    class_catalog: Dict[str, Dict[str, str]]
    has_scaler: bool
