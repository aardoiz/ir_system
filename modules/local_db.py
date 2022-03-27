from typing import List, Union
import pickle
from modules.models.text_process import eval_preproces
from sentence_transformers import SentenceTransformer, util
from time import time


with open('eval/data/datos_sqac.pkl', 'rb') as f:
    data = pickle.load(f)

with open('eval/data/embbedings.pkl', 'rb') as fl:
    emb = pickle.load(fl)

sentence_transformers_model = ("eduardofv/stsb-m-mt-es-distiluse-base-multilingual-cased-v1")
model = SentenceTransformer(sentence_transformers_model)


def Get_local_data() -> Union[List[str], List[str], List[str], List[List]]:
    """
    Use cursor to store data from db in python lists and use it elsewhere.
    """

    list_of_documents = []
    list_of_paragraphs = []
    list_of_sentences = []
    list_of_embeddings = []
    list_of_questions = []

    for i,document in enumerate(data):
        list_of_documents.append(i)
        list_of_paragraphs.append(eval_preproces(document["title"]))
        list_of_sentences.append(eval_preproces(document["content"]))
        list_of_embeddings.append(emb[i])

        list_of_questions.append(document["question"])

    return list_of_documents, list_of_paragraphs, list_of_sentences, list_of_embeddings, list_of_questions

