from typing import List, Tuple

from modules.utils.pickle_loader import load_local, load_sample
from modules.utils.text_process import eval_preproces

# Load the sample data
data = load_sample()
print(len(data))
# Load the real data stored in pickle folder
try:
    #data = load_local()
    print("USING LOCAL DATA")

except Exception as err:
    print("USING SAMPLE DATA")


def get_local_data() -> Tuple[List[str], List[str], List[str], List[List]]:
    """
    Use the preloaded data to store its info in python lists and use it on the main script.
    """

    list_of_documents = []
    list_of_titles = []
    list_of_sentences = []
    list_of_questions = []

    for i, document in enumerate(data):
        list_of_documents.append(i)
        list_of_titles.append(document["title"])
        list_of_sentences.append(eval_preproces(document["content"]))
        try:
            list_of_questions.append(document["question"])
        except:
            list_of_questions.append([])

    return list_of_documents, list_of_titles, list_of_sentences, list_of_questions
