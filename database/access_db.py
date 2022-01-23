from pymongo import MongoClient
from torch import Tensor


cluster = "mongodb+srv://aardoiz:M0nG0!_@cluster0.wx3m9.mongodb.net/myFirstDatabase?retryWrites=true&w=majority"
# conectamos a mongo
client = MongoClient(cluster)

cursor = client.tfm_data.apuntes.find({})
def Get_all_data():

    list_of_documents = []
    list_of_paragraphs = []
    list_of_sentences = []
    list_of_embeddings = []

    for document in cursor:
        list_of_documents.append(document['document'])
        list_of_paragraphs.append(document['paragraph'])
        list_of_sentences.append(document['sentence'])
        list_of_embeddings.append(document['embedding'])

    return list_of_documents , list_of_paragraphs , list_of_sentences ,list_of_embeddings

