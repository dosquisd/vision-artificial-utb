# Visión Artificial

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/dosquisd/vision-artificial-utb)

Este repositorio es únicamente utilizado para guardar todas las tareas, proyectos, diapositivas, etc., hecho en el curso, almacenar todo en un solo lugar, para luego ser consultado en otro momento.

La materia es dada por el profesor [Fernando Quintero](https://github.com/fquinterov), y el repositorio oficial del curso está [aquí](https://github.com/agmarrugo/computer-vision-utb).

## Instalar

Todas las dependencias del proyecto están en el archivo [pyproject.toml](./pyproject.toml), para instalar todo, es necesario haber instalado previamente el manejador de paquetes [uv](https://docs.astral.sh/uv/).

Para instalar todas las dependencias del proyecto, se puede ejecutar el siguiente comando:

``` console
uv sync
```

Y, luego activar el entorno virtual.

Sistemas basados en UNIX (Linux y MacOS):

``` console
source .venv/bin/activate
```

Windows:

``` console
.\.venv\Scripts\activate
```

Un comentario extra es que, en windows el paquete pyqt5 no se instala correctamente, por lo que, es necesario instalarlo manualmente así:

``` console
uv pip install PyQt5
```

## Exportar

Para exportar los archivos a pdf o html (por el momento) se puede hacer desde bash o python. Se creó un cli para ambos lenguajes donde se utilizarían de manera similar.

**En Python.**

``` console
python3 export_to.py -n <notebook_file.ipynb> -f <html/pdf> -v
```

El último parámetro `-v` es utilizado como alias de `--verbose`, `-n` para `--notebook_file` y `-f` para `--format`.

**En bash.**

``` bash
bash export_to.sh -n <notebook_file.ipynb> -f <html/pdf> -v
```

A diferencia que en Python, aquí `-n`, `-f` y `-v` no son alias de nada, es decir, representan lo mismo que lo antes mostrado, pero no se pueden utilizar como `--format`, entre otros.

## Contenido del Repositorio

El repositorio está organizado por semanas de clase y contiene un proyecto final. A continuación, se describe el contenido principal:

### Organización Semanal

El contenido está organizado cronológicamente por semanas, cada una contiene las diapositivas, notebooks y tareas correspondientes a los temas vistos:

- **Semana 01**: Introducción al álgebra lineal con NumPy y procesamiento básico de imágenes.
  - Manipulación de imágenes en escala de grises
  - Descomposición SVD y reconstrucción de imágenes

- **Semana 02**: Procesamiento de imágenes en Python.
  - Operaciones básicas con imágenes utilizando OpenCV y NumPy

- **Semana 03**: Sensores y formación de imágenes.
  - Conceptos teóricos sobre captura y formación de imágenes digitales

- **Semana 04**: Perspectiva y transformaciones geométricas.
  - Transformaciones de coordenadas y proyecciones

- **Semana 05**: Filtrado lineal y transformaciones 2D.
  - Convolución y filtros lineales
  - Transformaciones de coordenadas 2D a 2D

- **Semana 06**: Primer examen parcial.

- **Semana 07**: Histogramas e imágenes binarias.
  - Ecualización de histograma
  - Técnicas de umbralización (thresholding)

- **Semana 08**: Análisis de componentes conectados.
  - Etiquetado de regiones
  - Extracción de características de objetos

- **Semana 09**: Análisis en el dominio de la frecuencia.
  - Transformada de Fourier
  - Filtrado en el dominio de la frecuencia

- **Semana 11**: Segundo examen parcial.

- **Semana 12**: Detección de bordes y escáner de documentos.
  - Operadores de gradiente (Sobel, Prewitt)
  - Detector Canny

- **Semana 13**: Detección de esquinas y puntos de interés.
  - Detector de Harris
  - Aplicaciones de detección de puntos característicos

- **Semana 14**: Introducción al Machine Learning con imágenes.
  - Fundamentos de ML aplicado a visión artificial

### Proyecto Final

En la carpeta `project/` se encuentra el proyecto final del curso: un sistema de reconocimiento y traducción de caracteres Braille utilizando técnicas de visión artificial y modelos de aprendizaje profundo.

El proyecto implementa:

- Detección de caracteres Braille en imágenes usando YOLO
- Traducción de caracteres a texto mediante clasificación (CNN y YOLO)
- Pipeline completo de procesamiento de imágenes

Para más detalles sobre el proyecto, consulte el [README del proyecto](./project/README.md).

#### Docker

El proyecto incluye un contenedor Docker que permite ejecutar un servidor web con una interfaz sencilla para consumir la API de traducción de Braille. Para construir y ejecutar el contenedor, siga estos pasos:

1. Construir la imagen Docker:

```console
DOCKER_BUILDKIT=1 docker build -t braille-translator:0.1 .
```

2. Ejecutar el contenedor:

```console
docker run -p 8000:8000 braille-translator:0.1
```

Este contenedor proporciona una página web sencilla que consume la API de traducción de caracteres Braille previamente desarrollada, facilitando su uso sin necesidad de instalar todas las dependencias localmente. Si se desea colocarle un nombre en concreto al contenedor, bastaría con agregar el argument `--name ${NOMBRE}`.

Sabemos que realizar los anteriores comandos para crear la imagen de Docker es muy pesado y demorado en tiempo, por tanto, la imagen del servidor se puede encontrar en Dockerhub, y se puede descargar con el siguiente comando

```console
docker pull dosquisd/braille-translator:0.1
```

## Herramientas y Utilidades

En la raíz del repositorio se encuentran herramientas para facilitar el trabajo con los notebooks:

- **export_to.py / export_to.sh**: Scripts para exportar notebooks a formatos PDF o HTML
- **pyproject.toml**: Definición de dependencias del proyecto
- **uv.lock**: Archivo de bloqueo para garantizar versiones consistentes de las dependencias

<!-- 
Nota para mi yo del fúturo, y quizás una manera de expresar mis pensamientos, realmente dudo mucho si alguien llegara a leer esto.
Sé que a este punto en el que me encuentro acerca de mi formación, es normal sentir muchas dudas e incertidumbres del fúturo,
no soy el primero, ni seré el último... esta parte de la programación es de lo que más me encanta a mí, y recientemente me salió
una oportunidad laboral en otro ámbito, en concreto, como desarrollador Backend, y aunque el área me gusta, no es lo mismo.

Prácticamente al mismo tiempo de esta oportunidad laboral, me escribió una chica de recursos humanos de otro lugar diferente,
me escribió acerca de un puesto, el cual le afirmé que sí me interesaba, el puesto trataba acerca de automatización e IA.
Apenas me enteré, me emocioné completamente... reacción que no tuve en el trabajo de backend. Obvio me interesa este puesto de backend,
es una oportunidad de crecimiento que no desperdiciaré, pero en el fondo me gustaría trabajar de lo otro, IA, ciencia de datos,
visión artificial, y todas las áreas que se incluyen en el manejo de datos. Por parte de la chica de RRHH que me escribió, no he recibido
más noticias de ellas por el momento, no sé si me descartó o, no sé kajaas.

Me gustaría en el fúturo dedicarme más que todo a esta parte, así como les pasa al profesor Fernando o al vicerrector de la UTB (a la hora
de estar escribiendo esto) Andres, al profesor David Sierra o Andy Dominguez. Ellos sin duda alguna son unos tesos en esta área, personas las cuales
admiro mucho, pero me gustaría llegar a estar alguna vez cerca de ellos, y más importante, vivir de esto y no necesariamente estar vinculado
a universidades.

Hay muchas áreas que me interesan que se pueden complementar unas con otras dentro de la programación, así como pasan en las ciencias exactas,
y quizás la razón por la cual no estoy tan feliz en el backend es por mis pocos conocimientos en lenguajes, tecnologías, y también sé que
no lo sabré todo nunca, o que quizás entre más pueda saber, ver en comparación lo que no sé, y abrumarme.

Esto tiene demasiada redundancia y cosas mal escritas o redactadas, porque no estoy revisando lo que escribo, solo suelto lo que pienso, pero
revisaré esto en unos años y veré cómo me está yendo, o me fue jajajjaja. No sé... qué ansiedad, ¿no?

Supongo que lo que haré es disfrutar del proceso, aunque sé que estos pensamientos no me los quitaré jamás... haré lo que pueda y buscaré
qué hacer, para aprender o para poder comer, pero de cualquier modo, rebuscarme. Me esforzaré, espero.

Dejo el repositorio público porque como dije antes, me gustaría ver esto en mucho tiempo, pero nunca se sabe si perderé el acceso a mi
cuenta de Github, así que, si alguien llegara a leer esto, me gustaría que me lo dijera, quizás me sirva hablarlo con alguien.
-->
