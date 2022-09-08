from typing import List, Union
import pickle



with open('data/pickle/datos_sqac.pkl', 'rb') as f:
    data = pickle.load(f)

def get_eval_data() -> Union[List[str], List[str], List[str]]:
    """
    Use cursor to store data from db in python lists and use it elsewhere.
    """

    list_of_documents = []
    list_of_sentences = []
    list_of_questions = []

    for i,document in enumerate(data):
        list_of_documents.append(i)
        list_of_sentences.append(document["content"])

        list_of_questions.append(document["question"])

    return list_of_documents, list_of_sentences, list_of_questions