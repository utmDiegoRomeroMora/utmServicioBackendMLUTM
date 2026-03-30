# Documentacion Oficial del Repositorio

Backend en FastAPI para inferencia de sequia usando un modelo Keras y artefactos serializados en PKL.

Este documento describe como instalar, configurar, ejecutar y desplegar el proyecto de forma reproducible.

## 1. Objetivo del proyecto

El servicio expone endpoints HTTP para:

- Verificar salud del servicio.
- Consultar metadatos del modelo.
- Predecir clase de sequia para una observacion individual.
- Predecir clase de sequia para multiples observaciones en lote.

La salida incluye clase predicha, confianza, nivel de severidad, descripcion y recomendacion.

## 2. Estructura del repositorio

```text
utmServicioBackendMLUTM/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ model_loader.py
│  └─ schemas.py
├─ datasets_finales/
│  └─ X_train_final.csv
├─ models/
│  ├─ modelo_entrenado.keras
│  └─ modelo_entrenado.pkl
├─ MLP.ipynb
├─ README.md
├─ render.yaml
├─ requirements.txt
└─ runtime.txt
```

## 3. Requisitos

- Python 3.11.10 (definido en runtime.txt).
- Pip actualizado.
- Modelo entrenado y artefactos en la carpeta models.

## 4. Artefactos requeridos

Para que la API pueda iniciar, deben existir:

- models/modelo_entrenado.pkl
- models/modelo_entrenado.keras

Nota tecnica:

- El archivo PKL debe contener al menos feature_cols y target_names.
- Si existe scaler en el PKL, sera usado durante la transformacion.

## 5. Instalacion local

### 5.1 Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 5.2 Linux o macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Ejecucion del servicio

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Servicio local:

- API base: http://localhost:8000
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 7. Variables de entorno

Variables soportadas por la API:

- MODEL_DIR: directorio de modelos. Valor por defecto: models
- MODEL_PKL: nombre del artefacto PKL. Valor por defecto: modelo_entrenado.pkl
- MODEL_KERAS: nombre del modelo Keras. Valor por defecto: modelo_entrenado.keras
- CORS_ALLOW_ORIGINS: lista CSV de origenes permitidos o *

Ejemplo en PowerShell:

```powershell
$env:MODEL_DIR = "models"
$env:MODEL_PKL = "modelo_entrenado.pkl"
$env:MODEL_KERAS = "modelo_entrenado.keras"
$env:CORS_ALLOW_ORIGINS = "http://localhost:3000,https://mi-frontend.com"
```

## 8. Flujo interno de inferencia

1. Al iniciar, startup_event carga PKL y modelo Keras en memoria.
2. En cada request, las features se alinean con feature_cols esperadas.
3. Si faltan columnas, se rellenan con 0.0.
4. Si el artefacto incluye scaler, se aplica transformacion.
5. El modelo Keras produce probabilidades por clase.
6. Se toma argmax como clase final y se construye la respuesta enriquecida.

## 9. Referencia oficial de API

### 9.1 GET /health

Descripcion:

- Verifica que el servicio este disponible.

Respuesta esperada:

```json
{
  "status": "ok"
}
```

### 9.2 GET /metadata

Descripcion:

- Devuelve metadatos del modelo y catalogo de clases.

Campos relevantes de respuesta:

- model_file
- feature_cols
- target_names
- objective
- class_catalog
- has_scaler

### 9.3 POST /predict

Descripcion:

- Ejecuta inferencia para una sola observacion.

Body de ejemplo:

```json
{
  "features": {
    "feature_1": 10.2,
    "feature_2": 0.5,
    "feature_3": 99.0
  },
  "return_proba": true
}
```

Respuesta de ejemplo:

```json
{
  "predicted_class": "D1",
  "predicted_index": 2,
  "confidence": 0.84,
  "severity_level": "sequia moderada",
  "severity_description": "Deficit de humedad con afectacion inicial en cultivos y abasto.",
  "recommendation": "Aplicar medidas de ahorro y priorizar usos esenciales del recurso.",
  "possible_mlp_results": ["Sin sequia", "D0", "D1", "D2", "D3", "D4"],
  "probabilities": {
    "Sin sequia": 0.04,
    "D0": 0.06,
    "D1": 0.84,
    "D2": 0.05,
    "D3": 0.01,
    "D4": 0.0
  }
}
```

### 9.4 POST /predict_batch

Descripcion:

- Ejecuta inferencia para multiples observaciones.

Body de ejemplo:

```json
{
  "observations": [
    {
      "feature_1": 10.2,
      "feature_2": 0.5,
      "feature_3": 99.0
    },
    {
      "feature_1": 7.1,
      "feature_2": 0.2,
      "feature_3": 87.0
    }
  ],
  "return_proba": false
}
```

Respuesta:

- Lista de objetos con el mismo formato de POST /predict.

## 10. Pruebas rapidas con PowerShell

### 10.1 Health

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

### 10.2 Metadata

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/metadata" -Method Get | ConvertTo-Json -Depth 8
```

### 10.3 Predict

```powershell
$body = @{
  features = @{
    feature_1 = 10.2
    feature_2 = 0.5
    feature_3 = 99.0
  }
  return_proba = $true
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 10
```

### 10.4 Predict Batch

```powershell
$batchBody = @{
  observations = @(
    @{ feature_1 = 10.2; feature_2 = 0.5; feature_3 = 99.0 },
    @{ feature_1 = 7.1; feature_2 = 0.2; feature_3 = 87.0 }
  )
  return_proba = $false
} | ConvertTo-Json -Depth 8

Invoke-RestMethod -Uri "http://localhost:8000/predict_batch" -Method Post -Body $batchBody -ContentType "application/json" | ConvertTo-Json -Depth 10
```

## 11. Despliegue oficial en Render

### 11.1 Opcion recomendada: render.yaml (Blueprint)

1. Publica el repositorio en GitHub.
2. En Render, crea un Blueprint.
3. Selecciona el repositorio.
4. Render aplicara la configuracion declarada en render.yaml.

### 11.2 Variables sugeridas en produccion

- PYTHON_VERSION=3.11.10
- TF_CPP_MIN_LOG_LEVEL=3
- OMP_NUM_THREADS=1
- WEB_CONCURRENCY=1
- MODEL_DIR=models
- MODEL_PKL=modelo_entrenado.pkl
- MODEL_KERAS=modelo_entrenado.keras

## 12. Solucion de problemas

- Error de columnas faltantes:
  - Consulta GET /metadata y envia exactamente los nombres en feature_cols.
- Error cargando modelo Keras:
  - Verifica que el archivo exista en models y sea compatible con la version de TensorFlow/Keras.
- Consumo alto de memoria en Render:
  - Usa un worker unico y reduce tamano del modelo si es necesario.

## 13. Buenas practicas

- Mantener versionado separado para codigo y pesos grandes del modelo.
- Proteger endpoints con capa de autenticacion cuando se exponga publicamente.
- Registrar logs de inferencia sin exponer datos sensibles.

## 14. Licencia y autoria

Define aqui la licencia del proyecto (por ejemplo MIT) y los responsables del mantenimiento.
