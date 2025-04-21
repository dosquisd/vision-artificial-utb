# Models

El modelo base que se utilizó para el modelo de YOLO, es [yolo11n.pt](./yolo11n.pt). Descargado directamente del siguiente link: [https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt).

Existe una carpeta llamada [runs](./runs/) donde se muestra el procesamiento de entrenamiento y validación del modelo. Esta es una carpeta autogenerado por YOLO.

Originalmente, la carpeta [train2](./runs/detect/train2/) fue ejecutada por aparte, sin haber configurado correctamente el archivo [settings.json](~/.config/Ultralytics/settings.json), y luego fue movida a donde está ahora. Las carpetas [train](./runs/detect/train/) y [train3](./runs/detect/train3/) fueron las resultantes del proceso de entrenamiento y evaluación de un mismo modelo, respectivamente. Para estas últimas, el archivo [settings.json](~/.config/Ultralytics/settings.json) ya fue configurado como es debido. Se ve algo tal que así:

``` json
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
