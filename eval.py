import numpy as np
import pickle

from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from torch import cuda, device, tensor, topk
from typing import List

from modules.eval_db import get_eval_data
from modules.utils.text_process import e_preprocess, cross_preprocess

from time import time


device = device("cuda" if cuda.is_available() else "cpu")
print(f"Device selected: {device}")

all_documents_, all_sentences_, questions_eval = get_eval_data()

cross_encoder_model = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
cross = CrossEncoder(cross_encoder_model, device=device)

# Okapi BM25
tokenized_corpus_sentence = [e_preprocess(doc).split(" ") for doc in all_sentences_]
bm25_sentence = BM25Okapi(tokenized_corpus_sentence)


def compute_bm(search: str) -> dict:
    """
    This method does the following steps:
    - Process the query to get all keywords
    - Compare this list of keywords with pre-processed corpus sentences and paragraphs.
    - Use BM25 to compute the score between the query and each sentence.
    - Returns the top-7.
    """
    tokenized_query = e_preprocess(search).split(" ")

    doc_scores = tensor(np.array(bm25_sentence.get_scores(tokenized_query)))
   
    best = topk(doc_scores, 20)
    output = []
    for score, idx in zip(best[0], best[1]):
        idx = int(idx)
        score = round(float(score), 4)
        if score == 0.0:
            continue
        output.append(all_documents_[idx])

    return output


def compute_crossencoder(search: str) -> dict:
    """
    This method does the following steps:
    - Process the query to get all keywords
    - Compare this list of keywords with pre-processed corpus sentences and paragraphs.
    - Use BM25 to compute the score between the query and each sentence.
    - Return the top-7 results.
    - Create a list of strings concatenating the query and each one of the BM25 results.
    - Compute the cross-encoder score for each concatenated string.
    - Re-arrange the order in base of the highest new scores.
    """
    tokenized_query = e_preprocess(search).split(" ")

    doc_scores = tensor(np.array(bm25_sentence.get_scores(tokenized_query)))
   
    best = topk(doc_scores, 20)
    output = []
    for score, idx in zip(best[0], best[1]):
        idx = int(idx)
        score = round(float(score), 4)
        if score == 0.0:
            continue
        output.append(
            {
                "Oración": all_sentences_[idx],
                "Documento": all_documents_[idx],
            },
        )

    combinations = [[cross_preprocess(search), cross_preprocess(sen["Oración"])] for sen in output]

    sim_score = cross.predict(combinations)
    sim_score_argsort = reversed(np.argsort(sim_score))


    real_out = []
    for idx in sim_score_argsort:
        real_out.append(output[idx]["Documento"])

    return real_out


class results_eval(BaseModel):
    index: int
    question: str
    bm_response: List[int]


"""
# Sacar los resultados para el BM25
resultados = []
for index, question in enumerate(questions_eval):
    question = question[1:-1] #quitamos '?' y '¿'
    ans = compute_bm(question)
    resultados.append(results_eval(index = index, question=question, bm_response=ans))

with open ('data/pickle/resultados_bm25_no_stops.pkl', 'wb') as f: 
    pickle.dump(resultados, f)
"""

inicio = time()
# Sacar los resultados para el Cross-encoder
resultados = []
for index, question in enumerate(questions_eval):
    if index%250 == 0:
        print(index)
        print(inicio-time())
    question = question #quitamos '?' y '¿'
    ans = compute_crossencoder(question)
    resultados.append(results_eval(index = index, question=question, bm_response=ans))

with open ('data/pickle/resultados_cross_base.pkl', 'wb') as f:
    pickle.dump(resultados, f)