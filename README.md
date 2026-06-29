# 🌱 Plant Health MVP — Clasificación de Salud Vegetal (sano / enfermo)

MVP de IA aplicada para monitoreo agroindustrial. Analiza **imágenes o video** de hojas
de cultivo, detecta automáticamente si la planta está **sana o enferma** y expone el
resultado mediante una **API REST** que devuelve un JSON estructurado, consumible por
otros sistemas de la compañía.

El objetivo de negocio es **reducir la revisión visual manual** en procesos
agroindustriales, eliminando la subjetividad entre evaluadores y habilitando trazabilidad
y escalabilidad.

---

## 📋 Tabla de contenidos

- [Problema y enfoque](#-problema-y-enfoque)
- [Dataset](#-dataset)
- [Arquitectura](#-arquitectura)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Instalación](#-instalación)
- [Entrenamiento](#-entrenamiento)
- [Inferencia desde terminal](#-inferencia-desde-terminal)
- [API REST](#-api-rest)
- [Ejecución con Docker](#-ejecución-con-docker)
- [Variables de entorno](#-variables-de-entorno)
- [Seguimiento de experimentos (MLflow)](#-seguimiento-de-experimentos-mlflow)
- [Resultados](#-resultados)
- [Limitaciones](#-limitaciones)

---

## 🎯 Problema y enfoque

Parte de la validación visual del producto agrícola se realiza manualmente, lo que genera
diferencias entre evaluadores, baja trazabilidad y poca capacidad de escalar.

Este MVP aborda el enfoque de **detección de condición sana / no sana** mediante un
clasificador de imágenes. Se parte del dataset multiclase **PlantVillage** y se colapsa a
un problema binario: cualquier carpeta cuyo nombre contiene `healthy` se etiqueta como
*sano* (0); el resto, como *enfermo* (1).

La elección de un problema binario responde directamente a la pregunta de negocio: lo que
la compañía necesita primero es saber **si una planta requiere o no atención**, no
necesariamente qué enfermedad específica tiene.

---

## 📊 Dataset

**[PlantVillage](https://www.kaggle.com/code/tanishqraina/plantvillage/input)**
(Kaggle) — ~54.000 imágenes de hojas, 38 clases (combinaciones de cultivo + condición),
14 especies. Imágenes capturadas en condiciones controladas (fondo uniforme, una hoja por
imagen).

El dataset **no se incluye** en el repositorio. Para reproducir:

1. Descárgalo de Kaggle y descomprímelo.
2. Colócalo de modo que exista una carpeta con las subcarpetas de clases:

```
data/PlantVillage/
├── Apple___healthy/
├── Apple___Apple_scab/
├── Tomato___healthy/
├── Tomato___Late_blight/
└── ... (38 carpetas)
```

3. Ajusta la variable `DATA_DIR` en tu `.env` para que apunte a esa carpeta.

> **Notas sobre variantes del dataset:**
> - Otras versiones ya vienen divididas en `train/` y `val/`. El pipeline genera su **propio
>   split estratificado** (train/val/test) a partir de una sola carpeta, por lo que basta
>   apuntar `DATA_DIR` a `train/`.

---

## 🏗 Arquitectura

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────┐
│   Cliente   │────▶│   API REST   │────▶│  Capa de modelo  │
│ imagen/video│     │   FastAPI    │     │ EfficientNet-B0  │
└─────────────┘     └──────┬───────┘     └──────────────────┘
                           │
                    ┌──────▼────────┐
                    │ Reglas negocio │  umbral de confianza
                    │  (separadas)   │  + política de decisión
                    └───────────────┘
```

El diseño respeta una **separación estricta de responsabilidades**:

| Capa | Responsabilidad | Archivo |
|------|-----------------|---------|
| **Modelo** | Clasifica: dice *qué* (sano/enfermo) y con qué confianza | `src/model/inference.py` |
| **Reglas de negocio** | Decide *qué hacer* con la predicción (aprobar, marcar, revisar) | `src/business/rules.py` |
| **API** | Orquesta HTTP: recibe el archivo, responde el JSON | `src/api/main.py` |

Esta separación permite cambiar la política de negocio (umbrales, acciones) sin tocar el
modelo ni la API.

**Modelo:** EfficientNet-B0 con *transfer learning* (pesos preentrenados en ImageNet). Se
elige por su excelente relación precisión/costo, lo que lo hace liviano para inferencia,
contenedores y escenarios *edge*. El entrenamiento usa una estrategia de *warmup* con el
backbone congelado, seguida de *fine-tuning* completo.

**Video:** se procesa como una secuencia de *frames* muestreados; cada frame se clasifica
con el mismo modelo y los resultados se agregan (ratio de frames enfermos + voto
mayoritario).

---

## 📁 Estructura del repositorio

```
plant-health-mvp/
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .env.example
├── README.md
└── src/
    ├── config.py              # configuración centralizada (variables de entorno)
    ├── api/
    │   ├── main.py            # app FastAPI: endpoints /predict y /health
    │   └── schemas.py         # esquemas Pydantic v2 (unión discriminada)
    ├── business/
    │   └── rules.py           # capa de reglas de negocio (separada del modelo)
    └── model/
        ├── architecture.py    # definición de EfficientNet-B0
        ├── dataset.py         # colapso binario, split estratificado, transforms
        ├── train.py           # entrenamiento + evaluación + logging MLflow
        └── inference.py       # inferencia de imagen y video (carga única del modelo)
```

---

## ⚙️ Instalación

Requiere **Python 3.10+**.

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd plant-health-mvp

# 2. Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Configurar variables de entorno
cp .env.example .env            # luego edita DATA_DIR según tu dataset
```

---

## 🚂 Entrenamiento

```bash
python -m src.model.train
```

El script:
1. Descubre las imágenes y aplica el colapso binario.
2. Genera un split estratificado train/val/test.
3. Compensa el desbalance de clases mediante *class weights*.
4. Entrena EfficientNet-B0 (warmup con backbone congelado → fine-tuning).
5. Selecciona el mejor modelo priorizando el **recall de la clase enfermo** (el error
   costoso del negocio es dejar pasar una planta enferma como sana, puede desencadenar
   un efecto dominó en las plantas sanas del lote).
6. Evalúa en el set de test y guarda métricas, matriz de confusión y el modelo.

**Salidas:**
- `models/best_model.pth` — modelo entrenado.
- `models/confusion_matrix.png` — matriz de confusión sobre test.
- `models/classification_report.txt` — reporte por clase.
- Registro completo en MLflow.

> **Entrenamiento en CPU:** el dataset completo es pesado para una máquina sin GPU. Usa la
> variable `MAX_PER_CLASS` (p. ej. `120`) para submuestrear y entrenar en ~minutos sin
> sacrificar significativamente la precisión en este problema binario.

---

## 🔍 Inferencia desde terminal

Permite probar el modelo sobre un archivo individual (imagen o video):

```bash
# Imagen
python -m src.model.inference ruta/a/hoja.jpg

# Video
python -m src.model.inference ruta/a/video.mp4
```

Devuelve el JSON de predicción del modelo por consola.

---

## 🌐 API REST

Levantar el servicio en local:

```bash
uvicorn src.api.main:app --reload
```

Documentación interactiva (Swagger UI):

```
http://localhost:8000/docs
```

### Endpoints

| Método | Ruta | Descripción |
|--------|------|-------------|
| `GET`  | `/health`  | Verifica que el servicio está vivo y el modelo cargado |
| `POST` | `/predict` | Recibe una imagen o video y devuelve predicción + decisión |

### Ejemplo de consumo

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@hoja.jpg"
```

### Respuesta — imagen

```json
{
  "input_type": "image",
  "prediction": {
    "label": "diseased",
    "is_healthy": false,
    "confidence": 0.9942
  },
  "probabilities": {
    "healthy": 0.0058,
    "diseased": 0.9942
  },
  "business_decision": {
    "action": "flag_for_review",
    "reason": "Condición no saludable detectada con confianza suficiente",
    "threshold_applied": 0.85
  },
  "metadata": {
    "model_version": "efficientnet-b0",
    "inference_time_ms": 47.3
  }
}
```

### Respuesta — video

```json
{
  "input_type": "video",
  "frames_analyzed": 45,
  "frame_interval": 15,
  "summary": {
    "verdict": "diseased",
    "is_healthy": false,
    "diseased_ratio": 0.73,
    "avg_confidence": 0.89
  },
  "per_class_frames": { "healthy": 12, "diseased": 33 },
  "metadata": {
    "model_version": "efficientnet-b0",
    "inference_time_ms": 1240.5,
    "total_frames": 680,
    "fps": 30.0
  }
}
```

### Decisiones de negocio posibles

| Acción | Condición |
|--------|-----------|
| `approve` | Sano con confianza ≥ umbral |
| `flag_for_review` | Enfermo con confianza ≥ umbral |
| `manual_review` | Confianza < umbral (zona gris → escala a un humano) |

---

## 🐳 Ejecución con Docker

```bash
# Construir la imagen
docker build -t plant-health-api .

# Ejecutar el contenedor
docker run -p 8000:8000 plant-health-api
```

Luego abre `http://localhost:8000/docs`.

La imagen instala **PyTorch CPU-only** (evitando ~2-3 GB de dependencias CUDA innecesarias
para inferencia en CPU), corre como **usuario no-root** y excluye el dataset del contexto de
build mediante `.dockerignore`.

Para sobrescribir configuración en tiempo de ejecución:

```bash
docker run -p 8000:8000 -e CONFIDENCE_THRESHOLD=0.9 plant-health-api
```

> El modelo entrenado (`models/best_model.pth`) debe existir antes de construir la imagen,
> ya que se copia dentro del contenedor.

---

## 🔧 Variables de entorno

Configurables vía `.env` o `-e` en Docker. Ver `.env.example` para los valores por defecto.

| Variable | Descripción | Default |
|----------|-------------|---------|
| `DATA_DIR` | Carpeta con las subcarpetas de clases | `data/PlantVillage` |
| `IMAGE_SIZE` | Tamaño de imagen (px) | `224` |
| `MAX_PER_CLASS` | Máx. imágenes por carpeta (0 = todas) | `0` |
| `BATCH_SIZE` | Tamaño de batch | `32` |
| `EPOCHS` | Épocas de entrenamiento | `15` |
| `LEARNING_RATE` | Tasa de aprendizaje | `0.001` |
| `FREEZE_EPOCHS` | Épocas con backbone congelado | `3` |
| `MODEL_DIR` / `MODEL_FILENAME` | Ubicación del modelo | `models` / `best_model.pth` |
| `MLFLOW_TRACKING_URI` | Backend de tracking de MLflow | `sqlite:///mlflow.db` |
| `CONFIDENCE_THRESHOLD` | Umbral de decisión de negocio | `0.85` |

---

## 📈 Seguimiento de experimentos (MLflow)

Cada entrenamiento registra automáticamente parámetros, métricas por época, métricas de
test y artefactos (matriz de confusión, classification report).

Para visualizar la interfaz:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Abre `http://localhost:5000`. En MLflow 3.x

---

## 📊 Resultados

Métricas sobre el **set de test** (datos no vistos durante el entrenamiento):

| Métrica | Valor |
|---------|-------|
| Accuracy | 0.974 |
| F1-score | 0.980 |
| ROC-AUC | 0.998 |
| Precision (enfermo) | 1.00 |
| Recall (enfermo) | 0.966 |
| Recall (sano) | 0.991 |

El alto **recall de la clase enfermo** es el resultado clave para el negocio: el modelo casi
no deja pasar plantas enfermas como sanas, que es el error más costoso en este contexto.

---

## ⚠️ Limitaciones

- **Dominio de laboratorio:** PlantVillage contiene imágenes en condiciones controladas
  (fondo uniforme, una hoja). En campo real (fondo complejo, iluminación variable, oclusión, ruido
  visual, etc) el rendimiento esperado sería menor. Una iteración futura debería incluir *fine-tuning* 
  con imágenes de campo.
- **Clasificación binaria:** el MVP responde "sano/enfermo".
- **Agregación de video simple:** el muestreo por intervalo fijo y el voto mayoritario son
  suficientes para el MVP, pero podrían refinarse (p. ej. ponderación temporal, detección de
  zonas afectadas).

---