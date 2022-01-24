# TFM
Autor: Alfonso Ardoiz Galaz

Máster: Letras Digitales UCM

Título: Construcción de un SRI de dominio general

## Pasos a seguir
0. Crear un entorno virtual nuevo y activarlo

1. Clonar este repositorio y colocarse en su carpeta principal
```
git clone git@github.com:aardoiz/ir_system.git
```

2. Instalar poetry
```
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

3. Una vez instalado poetry inicializarlo usando poetry install
```
poetry install
```

4. Para probar el buscador, hay que ejecutar el archivo api.py con pyhton
```
python3 api.py
```

5. Abrir el navegador y escribir lo siguiente para ir al front temporal (hasta que prepare algo más profesional):
```
http://localhost:8425/docs
```

6. En el front vemos que hay 4 métodos: 1 GET, y 3 POST. Cada uno de los post es un tipo de buscador. Para acceder a ellos, hay que pinchar encima y se abrirá un panel verde. En primer lugar hay una explicación del método; y para probarlo hay que pulsar el botón de "try it out".
Al pulsarlo se nos abrirá un espacio donde escribir con una "query" de prueba; simplemente tenemos remplazar "string" por lo que queramos buscar. Los resultados aparecen más abajo en la página debajo de "code" y "details".

## Datos actuales
En el buscador de momento solo están cargados los apuntes de las asignaturas de Edición y OIM del máster.


## To Do List

- [x] Refactorizar el código
- [x] Ajustar las dependencias con la base de datos
- [x] Preparar los diagramas del proyecto
- [x] Creación de un front temporal para pruebas sencillas
- [ ] Dockerizar el back
- [ ] Creación de un front profesional
- [x] Documentación de uso de back (este Readme)
- [x] Documentación a nivel de código
- [ ] Documentación de cara al front profesional
