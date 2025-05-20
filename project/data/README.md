# Datos del Proyecto

## Fuentes de Datos

Los datos utilizados en este proyecto provienen de múltiples fuentes:

1. **Conjunto de datos principal**: Obtenido de [SciDB Braille Dataset](https://www.scidb.cn/en/detail?dataSetId=b1df1a601acc47a6984aafa8f3ab8e92). Específicamente, se utilizaron las imágenes de `character_label/voc-data`. Estos fueron importados a la carpeta [raw](./raw/) manteniendo la estructura de conjuntos de `train` y `test` original.

2. **Conjunto de datos de Kaggle**: También se incorporaron datos del [Kaggle Braille Character Dataset](https://www.kaggle.com/datasets/shanks0465/braille-character-dataset), ubicados en [processed/kaggle](./processed/kaggle/).

3. **Caracteres individuales**: El proyecto incluye un extenso conjunto de imágenes de caracteres Braille individuales en [processed/character](./processed/character/).

## Estructura de Datos

Los datos están organizados en la siguiente estructura:

- **[raw/](./raw/)**: Datos originales sin procesar
  - **character/**: Imágenes de caracteres Braille sin procesar
  - **test/**: Conjunto de datos para pruebas
  - **train/**: Conjunto de datos para entrenamiento

- **[processed/](./processed/)**: Datos procesados para entrenamiento y evaluación
  - **character/**: Imágenes de caracteres individuales procesados
  - **kaggle/**: Datos procesados de Kaggle, incluyendo versión YOLO
  - **test/**: Datos de prueba procesados
  - **train/**: Datos de entrenamiento procesados

## Formatos de Anotación

Las anotaciones originales utilizan el formato PascalVOC (.xml), compatible con herramientas como `labelImg`. Para los modelos YOLO, es necesario convertir estas anotaciones al formato YOLO (.txt). Esta conversión se realiza usando el script [xml_to_txt_yolo.py](./xml_to_txt_yolo.py).

### Uso de la herramienta de conversión

Para convertir anotaciones de PascalVOC a YOLO:

```bash
python3 xml_to_txt_yolo.py
```

### Edición manual de anotaciones

Si necesita revisar o corregir manualmente las anotaciones, puede utilizar la herramienta `labelImg`:

```bash
labelImg ./data
```

O ser más específico con los directorios:

```bash
labelImg ./data/raw/train
```
