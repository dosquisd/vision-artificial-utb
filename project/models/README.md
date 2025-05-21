# Modelos del Proyecto

Este directorio contiene los modelos entrenados y resultados de entrenamiento para el sistema de traducción de Braille.

## Modelos Base

El proyecto utiliza modelos base de YOLO tanto para la detección como para la clasificación de caracteres:

- **[yolo11n.pt](./yolo11n.pt)**: Modelo base de YOLO para detección, utilizado en la versión v1 del modelo de detección de caracteres Braille
- **[yolo11n-cls.pt](./yolo11n-cls.pt)**: Modelo base de YOLO para clasificación, utilizado en la versión v2 del módulo de traducción

## Estructura de Modelos

Los modelos entrenados y los resultados de entrenamiento se organizan en la siguiente estructura:

- **[runs/](./runs/)**: Directorio autogenerado por YOLO que contiene resultados de entrenamiento
  - **[detect/](./runs/detect/)**: Resultados de modelos de detección
    - **train/**: Primer entrenamiento del modelo de detección
    - **train2/**: Segundo entrenamiento del modelo de detección
    - **train3/**: Evaluación del modelo de detección
  - **[translation/](./runs/translation/)**: Resultados de modelos de traducción
    - **trainX**: Modelos resultantes de la versión v1 de traducción
    - **train5-yolo/**: Entrenamiento del modelo YOLO para traducción
  - **[dcgan/](./runs/dcgan/)**: Resultados del entrenamiento DCGAN

## Historial de Entrenamiento

- Inicialmente, se entrenó un modelo en [train2](./runs/detect/train2/) antes de configurar correctamente el archivo de configuración de Ultralytics.
- Posteriormente, los entrenamientos en [train](./runs/detect/train/) y [train3](./runs/detect/train3/) corresponden al entrenamiento y evaluación de un mismo modelo, respectivamente, con la configuración correcta.
- Se pensó en la idea de crear un modelo para generación de caracteres Braille para la aumentación de datos, dicho modelo fue almacenado en [dcgan](./runs/dcgan/train/), pero este nunca fue utilizado.
- Luego, se empezó con el entrenamiento de la primera versión del modelo de traducción, almacenados en [train](./runs/translation/train/), [train2](./runs/translation/train2/), [train3](./runs/translation/train3/) y [train4](./runs/translation/train4/).
- El modelo de traducción basado en YOLO se entrenó y almacenó en [train5-yolo](./runs/translation/train5-yolo/).

## Configuración de Ultralytics

El proyecto utiliza la siguiente configuración para Ultralytics, especificada en `~/.config/Ultralytics/settings.json`:

```json
{
  "settings_version": "0.0.6",
  "datasets_dir": "....../project",
  "weights_dir": "....../project/models/weights",
  "runs_dir": "....../project/models/runs",
  "uuid": "......",
  "sync": true,
  "api_key": "",
  "openai_api_key": "",
  "clearml": true,
  "comet": true,
  "dvc": true,
  "hub": true,
  "mlflow": true,
  "neptune": true,
  "raytune": true,
  "tensorboard": true,
  "wandb": false,
  "vscode_msg": true
}
```

## Versiones de Modelos

El sistema implementa diferentes versiones de modelos que pueden seleccionarse en tiempo de ejecución:

1. **Modelos de Detección de Caracteres**:
   - Versión v1: Implementación YOLO para detectar caracteres Braille en imágenes.

2. **Modelos de Traducción**:
   - Versión v1: Modelo inicial para traducción de caracteres Braille.
   - Versión v2: Modelo basado en YOLO para clasificación de caracteres.
