# Backend FastAPI para modelo de sequia (Render)

Este proyecto expone un modelo Keras con artefactos en PKL mediante una API en FastAPI.

## 1. Estructura del proyecto

```text
utmServicioBackendMLUTM/
├─ app/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ model_loader.py
│  └─ schemas.py
├─ models/
│  └─ .gitkeep
├─ .gitignore
├─ render.yaml
├─ runtime.txt
├─ requirements.txt
└─ README.md
```

`runtime.txt` contiene:

```text
python-3.11.10
```

## 2. Artefactos que debes copiar

Desde tu notebook/entrenamiento, necesitas estos archivos:

- `modelo_entrenado.pkl`
- `modelo_entrenado.keras` (o el `.h5` real referido por `model_weights_path`)

Copialos en la carpeta `models/`:

```text
models/modelo_entrenado.pkl
models/modelo_entrenado.keras
```

## 3. Entorno local y ejecucion

### 3.1 Crear entorno virtual

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3.2 Ejecutar API local

```powershell
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3.3 Probar endpoints

Salud:

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get
```

Metadata (importante para saber columnas exactas):

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/metadata" -Method Get | ConvertTo-Json -Depth 8
```

Prediccion (ejemplo):

```powershell
$body = @{
  features = @{
    feature_1 = 10.2
    feature_2 = 0.5
    feature_3 = 99
  }
  return_proba = $true
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://localhost:8000/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json -Depth 8
```

Nota: Usa los nombres reales de columnas que devuelve `GET /metadata` en `feature_cols`.

## 4. Como funciona la carga del modelo

1. Se lee `models/modelo_entrenado.pkl`.
2. Se extraen `feature_cols`, `target_names`, `scaler`.
3. Se intenta cargar el modelo Keras en este orden:
   - `MODEL_KERAS` (por defecto `modelo_entrenado.keras`)
   - `model_weights_path` guardado dentro del PKL
   - fallback `models/modelo_entrenado.keras`
4. En prediccion:
   - Se alinean columnas segun `feature_cols`.
   - Faltantes se rellenan con `0.0`.
   - Se aplica `scaler.transform(...)` si existe.
   - Se obtiene `argmax` sobre probabilidades para clase final.

## 5. Deploy en Render

## Opcion A: usando `render.yaml` (recomendado)

1. Sube este repo a GitHub con los archivos del backend.
2. En Render: New + Blueprint.
3. Selecciona tu repositorio.
4. Render detecta `render.yaml` y crea el servicio web.

Configuracion usada:

- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 180 --graceful-timeout 30 --keep-alive 5`
- Env vars:
  - `PYTHON_VERSION=3.11.10`
  - `TF_CPP_MIN_LOG_LEVEL=3`
  - `OMP_NUM_THREADS=1`
  - `WEB_CONCURRENCY=1`
  - `MODEL_DIR=models`
  - `MODEL_PKL=modelo_entrenado.pkl`
  - `MODEL_KERAS=modelo_entrenado.keras`

## Opcion B: crear Web Service manual

Si no usas Blueprint, crea un Web Service Python y coloca exactamente:

- Build Command: `pip install -r requirements.txt`
- Start Command: `gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:$PORT`

Y define las mismas variables de entorno.

## 6. Recomendaciones importantes

- Verifica compatibilidad de versiones de TensorFlow/Keras entre entrenamiento e inferencia.
- Si `model_weights_path` del PKL apunta a una ruta local vieja, no importa si tienes `models/modelo_entrenado.keras` correcto.
- Si el repositorio se vuelve muy pesado, evita versionar los pesos y cargalos por otro mecanismo seguro.
- Render Free tiene 512 MB RAM: usar 1 worker reduce riesgo de OOM con TensorFlow.
- Si aun aparece OOM, considera un modelo mas ligero (por ejemplo TFLite) o subir a un plan con mas memoria.

## 7. Endpoint listos

- `GET /health`
- `GET /metadata`
- `POST /predict`

La documentacion interactiva queda en:

- `http://localhost:8000/docs` local
- `https://TU-SERVICIO.onrender.com/docs` en produccion
