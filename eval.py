import numpy as np
import torch
import pickle

from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from torch import cuda, device
from typing import List

from modules.local_db import Get_local_data
from modules.models.text_process import Preprocess


class Search(BaseModel):
    query: str

device = device("cuda" if cuda.is_available() else "cpu")
print(f"Device selected: {device}")


# Semantic Searcher

#ToDo: Arrglar esto

all_documents_, all_paragraphs_, all_sentences_, all_embedding_, questions_eval = Get_local_data()


sentence_transformers_model = (
    "eduardofv/stsb-m-mt-es-distiluse-base-multilingual-cased-v1"
)
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
model = SentenceTransformer(sentence_transformers_model, device=device)
cross = CrossEncoder(cross_encoder_model, device=device)

# Okapi BM25
tokenized_corpus_sentence = [Preprocess(doc).split(" ") for doc in all_sentences_]
tokenized_corpus_paragraph = [Preprocess(doc).split(" ") for doc in all_paragraphs_]

bm25_sentence = BM25Okapi(tokenized_corpus_sentence)
bm25_parragraph = BM25Okapi(tokenized_corpus_paragraph)


def compute_bm(search: Search) -> dict:
    """
    This method does the following steps:
    - Process the query to get all keywords
    - Compare this list of keywords with pre-processed corpus sentences and paragraphs.
    - Use BM25 to compute the score between the query and each sentence.
    - Returns the top-7.
    """
    tokenized_query = Preprocess(search).split(" ")

    doc_scores_sentences = bm25_sentence.get_scores(tokenized_query)
    doc_scores_sentences = np.array(doc_scores_sentences)
    # sentence_weigth = 1
    # doc_scores_sentences = doc_scores_sentences**sentence_weigth

    doc_scores_paragraphs = bm25_parragraph.get_scores(tokenized_query)
    doc_scores_paragraphs = np.array(doc_scores_paragraphs)
    # paragraph_weight = 1.2
    # doc_scores_paragraph = doc_scores_paragraph ** paragraph_weight

    # Aquí sacamos las score totales con la suma de los dos parámetros combinados
    doc_scores = torch.tensor(np.add(doc_scores_sentences, doc_scores_paragraphs))
    best = torch.topk(doc_scores, 20)
    output = []
    for score, idx in zip(best[0], best[1]):
        idx = int(idx)
        score = round(float(score), 4)
        if score == 0.0:
            continue
        output.append(all_documents_[idx])

    return output


def compute_crossencoder(search: Search) -> dict:
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
    tokenized_query = Preprocess(search).split(" ")

    doc_scores_sentences = bm25_sentence.get_scores(tokenized_query)
    doc_scores_sentences = np.array(doc_scores_sentences)
    # sentence_weigth = 1
    # doc_scores_sentences = doc_scores_sentences**sentence_weigth

    doc_scores_paragraphs = bm25_parragraph.get_scores(tokenized_query)
    doc_scores_paragraphs = np.array(doc_scores_paragraphs)
    # paragraph_weight = 1.2
    # doc_scores_paragraph = doc_scores_paragraph ** paragraph_weight

    # Aquí sacamos las score totales con la suma de los dos parámetros combinados
    doc_scores = torch.tensor(np.add(doc_scores_sentences, doc_scores_paragraphs))
    best = torch.topk(doc_scores, 20)
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

    combinations = [[search, sen["Oración"]] for sen in output]

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



# Sacar los resultados para el BM25
resultados = []
for index, question in enumerate(questions_eval):
    question = question[1:-1] #quitamos '?' y '¿'
    ans = compute_bm(question)
    resultados.append(results_eval(index = index, question=question, bm_response=ans))

with open ('eval/data/resultados_bm25.pkl', 'wb') as f:
    pickle.dump(resultados, f)


# Sacar los resultados para el Cross-encoder
resultados = []
for index, question in enumerate(questions_eval[:1]):
    question = question[1:-1] #quitamos '?' y '¿'
    ans = compute_crossencoder(question)
    resultados.append(results_eval(index = index, question=question, bm_response=ans))

with open ('eval/data/resultados_cross.pkl', 'wb') as f:
    pickle.dump(resultados, f)