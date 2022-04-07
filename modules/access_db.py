from typing import List, Union

from pymongo import MongoClient

# You have to change the cluster and copy your cluster URL
cluster = "mongodb+srv://aardoiz:jGFwdKac4NWCXSv1@cluster0.wx3m9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
# conectamos a mongo
client = MongoClient(cluster)

cursor = client.tfm_data.apuntes.find({})


def Get_data_from_db() -> Union[List[str], List[str], List[str], List[List]]:
    """
    Use cursor to store data from db in python lists and use it elsewhere.
    """

    list_of_documents = []
    list_of_paragraphs = []
    list_of_sentences = []
    list_of_embeddings = []

    for document in cursor:
        list_of_documents.append(document["document"])
        list_of_paragraphs.append(document["paragraph"])
        list_of_sentences.append(document["sentence"])
        list_of_embeddings.append(document["embedding"])

    return list_of_documents, list_of_paragraphs, list_of_sentences, list_of_embeddings
