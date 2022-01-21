import pymongo
import json


client = pymongo.MongoClient()

my_db = client["tfm_prueba"]

my_col = my_db["edicion"]

with open("data/json/OIM_Anotación.json", "rb") as f:
    data = json.load(f)
