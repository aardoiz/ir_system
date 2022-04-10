import pickle
from pymongo import MongoClient

# You have to change the cluster and copy your cluster URL
cluster = "mongodb+srv://zoro:roronoa@cluster0.vxdvg.mongodb.net/TFM"
# conectamos a mongo
client = MongoClient(cluster)

db = client.TFM.DocumentList

with open("data/pickle/document_list.pkl", "rb") as f:
    docs = pickle.load(f)

result = db.insert_many(docs)
