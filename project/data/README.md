# Data

Todos los datos fueron obtenidos de: [https://www.scidb.cn/en/detail?dataSetId=b1df1a601acc47a6984aafa8f3ab8e92](https://www.scidb.cn/en/detail?dataSetId=b1df1a601acc47a6984aafa8f3ab8e92). En concreto, solo se están tomando las imagenes de `character_label/voc-data`. Los datos fueron directamente importados a la carpeta [raw](./raw/), teniendo en cuenta las distinciones entre `train` y `test` que ahí se mencionan.

El formato de las anotaciones de las imagenes fue PascalVOC, lo cual significa que utilizaron .xml, formato el cual también soporta `labelImg`. Realmente, no debería hacer falta utilizar `labelImg`, pero en caso de hacer una corrección, se puede ejecutar perfectamente con el comando `labelImg ./data`, o ser más especifico con los directorios si así se desea.

## XML

Probablemente sea necesario hacer la conversión en los formatos de guardado de las anotaciones, de PascalVOC (.xml) a YOLO (.txt), por tanto, está el script [xml_to_txt_yolo.py](./xml_to_txt_yolo.py) para facilitar la tarea.
