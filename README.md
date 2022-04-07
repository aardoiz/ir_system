# TFM
Autor: Alfonso Ardoiz Galaz

Máster: Letras Digitales UCM

Título: Sistema de Recuperación de documentos personalizado. [[[Versión SQAC]]]

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

3. Poetry sirve para poder instalar las dependencias de este proyecto
```
poetry install
```

4. Para iniciar el buscador, hay que ejecutar el archivo api.py con pyhton
```
python3 api.py
```

5. El front del programa se puede acceder a través del navegador:
```
http://localhost:8425/app
```

6. En el front vemos que hay una caja de texto para introducir búsquedas (inicialmente aparece la oración: "lechugas") y dos botones de búsqueda: uno con el motor basado en BM25, y otro con el basado en Cross-encoders. Los resultados aparecen abajo como la lista de documentos relevantes y lista de los párrafos que contienen esas palabras. [WIP] [Introducir imagen]

## Esquema de funcionamiento

#### Proyecto
[WIP]

#### Núcleo Central (SRI)
[WIP]

#### Pre-procesado de documentos
En el preprocesamiento, hay dos módulos principales. El "parser" de archivos pdf y el de archivos html.

![Pre-procesamiento](data/img/Text_Parser.png?raw=true "Módulo de pre-procesado de documentos")

El output de cada "parser" es una lista de objectos "Document". Cada Document se compone de lo siguiente:
![Document](data/img/Document_Object.png?raw=true "Objeto Document")

"type" -> asignatura

"document" -> tema 

"embedding" -> representación numérica de la oración procesada usando S-BERT.


#### Base de datos - Local vs MongoDB
[WIP]

## Datos actuales
Actualmente se está usando la base de datos local, dónde está cargado el corpus usado para evaluar el sistema.

## To Do List

- [ ] Reestructurar el proyecto
- [ ] Refactorizar el código
- [x] Ajustar las dependencias con la base de datos
- [ ] Preparar los diagramas del proyecto
- [x] Creación de un front temporal del buscador
- [ ] Creación de un front para la ingesta de documentos
- [ ] Dockerizar el proyecto
- [ ] Creación de un front profesional
- [x] Documentación de uso de front (este Readme)
- [x] Documentación a nivel de código
- [ ] Documentación de cara al front profesional

