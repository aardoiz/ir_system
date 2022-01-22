from pymongo import MongoClient

import datetime

#url de la base de datos
cluster = "mongodb+srv://aardoiz:M0nG0!_@cluster0.wx3m9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
#conectamos a mongo
client = MongoClient(cluster)

print(client.list_database_names())

db = client.tfm_data

print(db.list_collection_names())

# añadir datos
"""
todo1 = {"documento":"guia01", "parrafo":"que diver es edición. Spoiler NO", "frase":"que diver es edición", "fecha": datetime.datetime.utcnow()}

todo2 = [{"documento":"guia01", "parrafo":"que diver es edición. Spoiler NO", "frase":"que diver es edición", "fecha": datetime.datetime.utcnow()}
,{"documento":"guia01", "parrafo":"que diver es edición. Spoiler NO", "frase":"Spoiler NO", "fecha": datetime.datetime.utcnow()}
,{"documento":"guia01", "parrafo":"wololo", "frase":"wololo", "fecha": datetime.datetime.utcnow()}]


todos = db.todos

# meter en databade "todos"

#result = todos.insert_one(todo1)
result = todos.insert_many(todo2)"""