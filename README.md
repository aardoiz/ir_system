# TFM
Autor: Alfonso Ardoiz Galaz

Máster: Letras Digitales UCM

Título: Sistema de Recuperación de Documentos en español, propuesta de una nueva herramienta e-learning.

## Pasos a seguir
0. (Hasta meter docker) Crear un entorno virtual nuevo y activarlo

1. Clonar este repositorio y colocarse en su carpeta principal
```
git clone git@github.com:aardoiz/ir_system.git
```

2. Instalar poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

3. Poetry sirve para poder instalar las dependencias de este proyecto, además para procesar documentos pdf hay que instalar con pip un paquete python
```
poetry install
pip install pdfminer.six
```

4. Para iniciar el buscador, hay que ejecutar el archivo api.py con pyhton. (Realizar únicamente uno de los tres comandos)
```
python3 api.py
python api.py
py api.py
```

5. Una vez iniciado se puede acceder a la interfaz del buscador a través del navegador:
```
http://localhost:8425/app
```

6. En el front vemos que hay una caja de texto para introducir búsquedas (inicialmente aparece la oración: "lechugas") y dos botones de búsqueda: uno con el motor basado en BM25, y otro con el basado en Cross-encoders. Los resultados aparecen abajo como la lista de documentos relevantes y lista de los párrafos que contienen esas palabras. 


7. Si queremos procesar documentos personales, hay que editar la linea **219** del archivo "document_parser.py", dónde hay que incluir la ruta de la carpeta de archivos que queremos procesar. Si queremos procesar una página web, simplemente hay que escribir su URL en la misma línea. (Por el momento el programa solo puede procesar archivos .pdf, archivos .pptx y archivos .html)
```
process_path("RUTA A CAMBIAR")
```

## Esquemas de funcionamiento

#### Proyecto
![Esquema general](data/img/esquema_tfm_final.png?raw=true "Esquema general que comprende todo el proyecto")

#### Sistema de Recuperación de Información
Búsqueda con el motor estadístico basado en el algoritmo BM25
![Esquema BM25](data/img/esquema_bm25.png?raw=true "Esquema del motor basado en BM25")

Búsqueda con el motor híbrido basado en BM25 y en redes neuronales con arquitectura Cross-Encoder
![Esquema Cross-Encoder](data/img/esquema_crossencoder.png?raw=true "Esquema del motor basado en BM25")

#### Procesado de documentos y almacenamiento de la información
En el procesamiento de documentos hay tres funciones principales para tres tipos de archivos:

![Pre-procesamiento](data/img/esquema_procesamiento_docs.png?raw=true "Módulo de procesado de documentos")

La salida de este procesamiento se almacena en la base de datos local, y los documentos procesados se mandan a una carpeta llamada "done".

Por último, se pueden almacenar los datos en nuestra cuenta de MongoDB, dotando al programa de portabilidad. Este paso es totalmente opcional.

## Datos actuales
Actualmente el programa usa la base de datos de muestra, dónde está cargado el corpus usado para evaluar el sistema.

## Interfaz de usuario
La interfaz de usuario se ha diseñado con especial cuidado para que sea fácil de entender y de usar. El usuario cuenta con un campo de texto en el que introducir las búsquedas, un botón para usar el buscador basado en el algoritmo BM25, otro botón para el buscador basado en Cross-Encoder; y dos botones más para ayudar al usuario. 
![Front](data/img/front_ejemplo.png?raw=true "Interfaz gráfica del buscador")

## To Do List

- [x] Reestructurar el proyecto
- [x] Refactorizar el código
- [x] Ajustar las dependencias con la base de datos
- [x] Preparar los diagramas del proyecto
- [x] Creación de un front temporal del buscador
- [ ] Dockerizar el proyecto
- [x] Creación de un front profesional
- [x] Documentación de uso de front (este Readme)
- [x] Documentación a nivel de código
- [x] Documentación de cara al front profesional

