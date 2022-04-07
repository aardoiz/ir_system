import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from torch import cuda, device

from modules.access_db import Get_data_from_db
from modules.local_db import Get_local_data
from modules.models.text_process import Preprocess, html_mark, get_regex

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--database', metavar='database', type=str, help="Select how to use the program")
args = parser.parse_args()
db = args.database

class Search(BaseModel):
    query: str


# FastAPI  Settings
app = FastAPI()
CORS_ORIGINS = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
device = device("cuda" if cuda.is_available() else "cpu")
print(f"Device selected: {device}")


if db == 'Mongo':
    all_documents_, all_paragraphs_, all_sentences_, all_embedding_ = Get_data_from_db()
else:
    all_documents_, all_paragraphs_, all_sentences_, all_embedding_, _ = Get_local_data()


sentence_transformers_model = (
    "eduardofv/stsb-m-mt-es-distiluse-base-multilingual-cased-v1"
)
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
model = SentenceTransformer(sentence_transformers_model)
cross = CrossEncoder(cross_encoder_model)

# Okapi BM25
tokenized_corpus_sentence = [Preprocess(doc).split(" ") for doc in all_sentences_]
tokenized_corpus_paragraph = [Preprocess(doc).split(" ") for doc in all_paragraphs_]

bm25_sentence = BM25Okapi(tokenized_corpus_sentence)
bm25_parragraph = BM25Okapi(tokenized_corpus_paragraph)


# Main methods
@app.get("/app")
def root():
    return FileResponse("./front/main.html")


@app.get('/favicon.ico')
async def favicon():
    return FileResponse("./front/favicon.ico")


@app.post("/bm25")
def compute_bm(search: Search) -> dict:
    """
    This method does the following steps:
    - Process the query to get all keywords
    - Compare this list of keywords with pre-processed corpus sentences and paragraphs.
    - Use BM25 to compute the score between the query and each sentence.
    - Returns the top-7.
    """
    tokenized_query = Preprocess(search.query).split(" ")

    regex = get_regex(tokenized_query)

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
    best = torch.topk(doc_scores, 7)
    output = []
    for score, idx in zip(best[0], best[1]):
        idx = int(idx)
        score = round(float(score), 4)
        if score == 0.0:
            continue
        output.append(
            {
                "Oración": all_sentences_[idx],
                "Párrafo": all_paragraphs_[idx],
                "Score": score,
                "Documento": all_documents_[idx],              
                "Oración_HTML": html_mark(all_sentences_[idx], regex),
                "Párrafo_HTML": html_mark(all_paragraphs_[idx], regex)
            }
        )

    out = {}
    out["Resultados"] = output
    return JSONResponse(content=out)


@app.post("/cross_encoder")
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
    tokenized_query = Preprocess(search.query).split(" ")
    regex = get_regex(tokenized_query)

    doc_scores_sentences = np.array(bm25_sentence.get_scores(tokenized_query))
    doc_scores_paragraphs = np.array(bm25_parragraph.get_scores(tokenized_query))


    # Aquí sacamos las score totales con la suma de los dos parámetros combinados
    doc_scores = torch.tensor(np.add(doc_scores_sentences, doc_scores_paragraphs))
    best = torch.topk(doc_scores, 7)
    output = []
    for score, idx in zip(best[0], best[1]):
        idx = int(idx)
        score = round(float(score), 4)
        if score == 0.0:
            continue
        output.append(
            {
                "Oración": all_sentences_[idx],
                "Párrafo": all_paragraphs_[idx],
                "Score": score,
                "Documento": all_documents_[idx],              
                "Oración_HTML": html_mark(all_sentences_[idx], regex),
                "Párrafo_HTML": html_mark(all_paragraphs_[idx], regex)
            },
        )

    combinations = [[search.query, sen["Oración"]] for sen in output]

    sim_score = cross.predict(combinations)
    sim_score_argsort = reversed(np.argsort(sim_score))

    count = 0
    real_out = []
    for idx in sim_score_argsort:
        count += 1
        real_out.append(
            {
                "Oración": output[idx]["Oración"],
                "Párrafo": output[idx]["Párrafo"],
                "Score": round(float(sim_score[idx]), 2),
                "Documento": output[idx]["Documento"],
                "Oración_HTML": html_mark(output[idx]["Oración"], regex),
                "Párrafo_HTML": html_mark(output[idx]["Párrafo"], regex)
            }
        )
        if count == 5:
            break

    out = {}
    out["Resultados"] = real_out
    # AHORA METEMOS CROSSENCODER

    return JSONResponse(content=out)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8425)
