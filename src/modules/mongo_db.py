from typing import List, Union
from pymongo import MongoClient

# You have to change the cluster and copy your cluster URL
cluster = "mongodb+srv://zoro:roronoa@cluster0.vxdvg.mongodb.net/TFM"
# conectamos a mongo
client = MongoClient(cluster)

cursor = client.TFM.DocumentList.find({})


def get_data_from_db() -> Union[List[str], List[str], List[str], List[List]]:
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
