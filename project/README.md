# Proyecto Visión Artificial

A final de semestre es necesario hacer entrega de un proyecto que se escoge desde finales del primer corte. En nuestro caso, es la primera vez que utilizamos modelos de IA en imágenes, así que realizaremos un proyecto que si bien no es novedoso, nos servirá bastante para aprender de todo este nuevo mundo que se nos está abriendo.

El proyecto a realizar es un traductor de Braille. La idea es que el modelo reciba una página completa escrita en Braille, y en base a eso, logre hacer la traducción a texto. A la hora de estar escribiendo este README, todavía no se han sacado los datos con los que se entrenarán los modelos, pero la idea es utilizar ingles-español únicamente.

Para cumplir con nuestra tarea, tenemos pensado utilizar dos modelos, ya sean preentrenados o no, aunque preferiblemente preentrenados, ya sea YOLO. Uno de los modelos podría encargarse de hacer el etiquetamiento de cada una de las celdas, sería un modelo de ML con aprendizaje no supervisado (el cual tampoco hemos utilizado nunca); y otro que se encargue de entender las celdas, este sí tiene que ser un modelo con apredizaje supervisado.

## Referencias

Como sabemos que no es algo nuevo, hemos consultado algunas referencias en los siguientes links:

* [https://paperswithcode.com/paper/optical-braille-recognition-using-object](https://paperswithcode.com/paper/optical-braille-recognition-using-object).
* [https://paperswithcode.com/dataset/dsbi](https://paperswithcode.com/dataset/dsbi).
* [https://ieeexplore.ieee.org/document/7065649](https://ieeexplore.ieee.org/document/7065649).
* [https://www.kaggle.com/datasets/shanks0465/braille-character-dataset](https://www.kaggle.com/datasets/shanks0465/braille-character-dataset).
* [https://www.scidb.cn/en/detail?dataSetId=b1df1a601acc47a6984aafa8f3ab8e92](https://www.scidb.cn/en/detail?dataSetId=b1df1a601acc47a6984aafa8f3ab8e92).
* [https://universe.roboflow.com/search?q=class%3Abraille](https://universe.roboflow.com/search?q=class%3Abraille).

## Estructura

La estructura inicial del proyecto sigue está distribuida de la siguiente manera:

1. [data/](./data/). Dentro de la carpeta está otras subcarpetas: [raw](./data/raw/) y [processed](./data/processed/). Primero está toda la información cruda dentro la primera subcarpeta, después de ser procesada, es almacenada en la última mencionada.
2. [notebooks/](./notebooks/). Dentro de esta carpeta, se encontrarán todas los archivos con extensión `.ipynb`, donde cada uno de los archivos tendrá un número prefijo indicando el orden en el que se han creado los archivos.
3. [src/](./src/). Está todo el código fuente para crear el proyecto. Hasta el momento, nada más tiene la carpeta [src/modeling/](./src/modeling/) donde iría el entrenamiento de datos.
