import pickle

from pymongo import MongoClient


# url de la base de datos
cluster = "mongodb+srv://aardoiz:M0nG0!_@cluster0.wx3m9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
# conectamos a mongo
client = MongoClient(cluster)
db = client.tfm_data

with open("data/pickle/document_list.pkl", "rb") as f:
    docs = pickle.load(f)

print(len(docs))
docs_dict = []
for element in docs:
    a = {}
    a["type"] = element.type
    a["document"] = element.document
    a["paragraph"] = element.paragraph
    a["sentence"] = element.sentence
    a["embedding"] = element.embedding.tolist()

    docs_dict.append(a)
base_datos = db.apuntes

result = base_datos.insert_many(docs_dict)
# añadir datos

print('You should delete pkl file')