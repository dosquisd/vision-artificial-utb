# Proyecto Visión Artificial - Traductor de Braille

Este proyecto implementa un sistema de reconocimiento y traducción de caracteres Braille utilizando técnicas de visión artificial y modelos de aprendizaje profundo. El sistema es capaz de detectar los caracteres Braille en imágenes, extraer cada carácter individual, y traducirlos a su equivalente textual.

## Descripción del Proyecto

El sistema implementa un pipeline completo de detección y traducción de Braille que consta de dos componentes principales:

1. **Detección de Caracteres**: Utiliza un modelo YOLO para localizar y detectar caracteres Braille en imágenes.
2. **Traducción de Caracteres**: Convierte cada carácter Braille detectado a su equivalente en texto utilizando un modelo de clasificación. Existen dos implementaciones:
   - Modelo CNN personalizado implementado en PyTorch
   - Modelo de clasificación YOLO

El proyecto implementa diferentes versiones de estos modelos y permite configurar su uso mediante el archivo principal `main.py`.

## Referencias

Como sabemos que no es algo nuevo, hemos consultado algunas referencias en los siguientes links:

- [Optical Braille Recognition using Object Detection](https://paperswithcode.com/paper/optical-braille-recognition-using-object)
- [DSBI Dataset](https://paperswithcode.com/dataset/dsbi)
- [IEEE Document on Braille Recognition](https://ieeexplore.ieee.org/document/7065649)
- [Kaggle Braille Character Dataset](https://www.kaggle.com/datasets/shanks0465/braille-character-dataset)
- [SciDB Braille Dataset](https://www.scidb.cn/en/detail?dataSetId=b1df1a601acc47a6984aafa8f3ab8e92)
- [Roboflow Braille Resources](https://universe.roboflow.com/search?q=class%3Abraille)
- [Ultralytics YOLOv8 Documentation](https://docs.ultralytics.com/)
- [PyTorch Documentation](https://docs.pytorch.org/docs/stable/index.html)

## Estructura del Proyecto

El proyecto está organizado en la siguiente estructura de directorios:

1. **[data/](./data/)**: Contiene los datos utilizados para entrenar y evaluar los modelos:
   - **[raw/](./data/raw/)**: Datos originales sin procesar
   - **[processed/](./data/processed/)**: Datos procesados para entrenamiento y evaluación
   - **xml_to_txt_yolo.py**: Script para convertir anotaciones de PascalVOC a formato YOLO

2. **[notebooks/](./notebooks/)**: Notebooks de Jupyter para exploración, entrenamiento y evaluación:
   - **001_exploring_dataset.ipynb**: Exploración inicial del conjunto de datos
   - **002_train_yolo_model.ipynb**: Entrenamiento del modelo YOLO para detección
   - **003_test_yolo.ipynb**: Evaluación del modelo YOLO
   - **004_train_dcgan_model.ipynb**: Entrenamiento del modelo DCGAN. No se utilizó de nada
   - **005_train_translation.ipynb**: Entrenamiento del modelo de traducción
   - **006_test_translation.ipynb**: Evaluación del modelo de traducción
   - **007_train_translation_yolo.ipynb**: Entrenamiento del modelo combinado
   - **008_test_translation_yolo.ipynb**: Evaluación del modelo combinado

3. **[models/](./models/)**: Contiene los modelos entrenados y resultados de entrenamiento:
   - **yolo11n.pt**: Modelo base de YOLO
   - **[runs/](./models/runs/)**: Resultados de entrenamiento de YOLO, DCGAN y traducción

4. **[src/](./src/)**: Código fuente del proyecto:
   - **[dataset/](./src/dataset/)**: Módulos para manejo y procesamiento de datos, incluye:
     - Extracción de caracteres individuales
     - Aplicación de filtros y preprocesamiento
     - Aumentación de datos para entrenamiento
     - Redimensionamiento y normalización de imágenes
   - **[modeling/](./src/modeling/)**: Implementación de modelos de aprendizaje profundo:
     - Modelos de traducción basados en CNN (PyTorch)
     - Modelos de detección y clasificación YOLO
   - **[versions/](./src/versions/)**: Diferentes versiones implementadas de los pipelines completos:
     - Versiones de detección de caracteres
     - Versiones de traducción de caracteres (v1: CNN, v2: YOLO)
   - **config.py**: Configuración general del proyecto con parámetros validados
   - **filter_characters.py**: Preprocesamiento de caracteres con técnicas de realce

5. **Archivos de configuración**:
   - **data.yaml**: Configuración para el modelo de detección de caracteres
   - **data_translation.yaml**: Configuración para el modelo de traducción
   - **main.py**: Punto de entrada principal que integra los componentes del sistema

## Uso

Para utilizar el sistema de traducción de Braille, ejecute el script principal:

```bash
python3 main.py
```

El script cargará los modelos preentrenados y procesará la imagen de prueba especificada en el código.

Para personalizar la entrada y los parámetros del modelo, modifique las variables en el bloque `__main__` del archivo `main.py`.

## Componentes Principales

El sistema está compuesto por dos componentes principales que pueden utilizarse en diferentes combinaciones:

### Detector de Caracteres

- **Versión v1**: Implementación basada en YOLO que detecta caracteres Braille individuales en la imagen.
- Configuración mediante los parámetros `conf` e `iou` para ajustar la sensibilidad de detección.

### Traductor de Caracteres

- **Versión v1**: Modelo CNN implementado en PyTorch que clasifica caracteres Braille individuales.
- **Versión v2**: Modelo de clasificación YOLO que procesa caracteres Braille y devuelve las predicciones con sus respectivos niveles de confianza.

Para seleccionar qué versión de cada componente utilizar, modifique los parámetros `version_translation` y `version_characters` en el script principal.
