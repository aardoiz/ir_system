from json import load
from typing import List, Tuple

from pymongo import MongoClient
from modules.utils.pickle_loader import load_local

# IMPORTANT!! You should change the cluster using your cluster URL
cluster = "mongodb+srv://<user>:<password>@cluster0.vxdvg.mongodb.net/TFM"

client = MongoClient(cluster)

# IMPORTANT!! You should change DatabaseName and CollectionName with your specific names
cursor = client.DatabaseName.CollectionName.find({})

db = client.TFM.DocumentList

def get_data_from_db() -> Tuple[List[str], List[str], List[str], List[List]]:
    """
    Use cursor to store data from db in python lists and use it elsewhere.
    """

    list_of_documents = []
    list_of_titles = []
    list_of_sentences = []

    for i, document in enumerate(cursor):
        list_of_documents.append(i)
        list_of_titles.append(document["title"])
        list_of_sentences.append(document["content"])

    return list_of_documents, list_of_titles, list_of_sentences


def upload_local_to_mongo():
    """
    Update the local data to MongoDB server to store your data and use it in other systems
    """
    docs = load_local()
    result = db.insert_many(docs)
