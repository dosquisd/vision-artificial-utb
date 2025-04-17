# Data

Todos los datos fueron obtenidos de: [https://www.scidb.cn/en/detail?dataSetId=b1df1a601acc47a6984aafa8f3ab8e92](https://www.scidb.cn/en/detail?dataSetId=b1df1a601acc47a6984aafa8f3ab8e92). En concreto, solo se están tomando las imagenes de `segment_label`, los `.json` los estamos ignorando porque no sabemos qué tanta compatibilidad haya con labelImg y YOLO.

En la carpeta [raw](./raw/) se encuentra puras imagenes de donde se sacaran los vectores respectivos para utilizar yolo finalmente. Dentro de esta carpeta están divididas entre `train` y `test` que son las imagenes usadas en entrenamiento y testing, respectivamente. Luego, en [processed](./processed/) se encuentra los vectores que luego son usados para YOLO, y como bien se mencionó antes, también está separado por `train` y `test`.

Los etiquetamientos se hacen utilizando el paquete de `labelImg`, utilizando por ejemplo, el comando: `labelImg ./data/raw/`, para luego cambiar el formato de guardado a YOLO, como se ve en la imagen de abajo. El nombre de la clase que se utilizará es "Character" siempre.

![labelImg-YOLO](../figures/labelimg_capture.png)

## XML

En caso de haber utilizado el formato de guardado XML o "PascalVOC", se puede utilizar el archivo [xml_to_txt_yolo.py](./xml_to_txt_yolo.py) para convertir los archivos que están guardados en formato XML a TXT en formato yolo.
