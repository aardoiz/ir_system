from typing import List, Union
import pickle
from modules.models.text_process import eval_preproces



with open('./eval/data/datos_sqac.pkl', 'rb') as f:
    data = pickle.load(f)


try:
    with open('./data/pickle/document_list.pkl', 'rb') as f:
        data = pickle.load(f)
        print('USING LOCAL DATA')

except Exception as err:
    print('USING SAMPLE DATA')


def Get_local_data() -> Union[List[str], List[str], List[str], List[List]]:
    """
    Use cursor to store data from db in python lists and use it elsewhere.
    """

    list_of_documents = []
    list_of_titles = []
    list_of_sentences = []
    list_of_questions = []

    for i,document in enumerate(data):
        list_of_documents.append(i)
        list_of_titles.append(document["title"])
        list_of_sentences.append(eval_preproces(document["content"]))
        try:
            list_of_questions.append(document["question"])
        except:
            list_of_questions.append([])

    return list_of_documents, list_of_titles, list_of_sentences, list_of_questions



with open('eval/data/datos_sqac_enhanced.pkl', 'rb') as fl:
    enhanced = pickle.load(fl)

def Get_enhanced_data()-> Union[List[str], List[str], List[str], List[List]]:
    """
    Use cursor to store data from db in python lists and use it elsewhere.
    """

    list_of_documents = []
    list_of_paragraphs = []
    list_of_sentences = []
    list_of_embeddings = []
    list_of_questions = []

    for i,document in enumerate(enhanced):
        for phrase in document["content"]:
            list_of_documents.append(i)
            list_of_paragraphs.append(eval_preproces(document["title"]))
            list_of_sentences.append(eval_preproces(phrase))
            list_of_embeddings.append([])

        list_of_questions.append(document["question"])

    return list_of_documents, list_of_paragraphs, list_of_sentences, list_of_embeddings, list_of_questions