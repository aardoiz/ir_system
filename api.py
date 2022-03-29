import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from torch import cuda, device

from modules.access_db import Get_data_from_db
from modules.local_db import Get_local_data
from modules.models.text_process import Preprocess


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


# Semantic Searcher

#ToDo: Arrglar esto
#all_documents_, all_paragraphs_, all_sentences_, all_embedding_ = Get_data_from_db()
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
@app.get("/")
def root():
    return {"message": "Hello"}


@app.post("/semantic_similarity")
def compute_sbert(search: Search) -> dict:
    """
    This method does the following steps:
    - Get the embedding of the input query using a S-BERT models.
    - Calculate the cosine distance between this embedding and all embeddings pre-computed from the corpus.
    - Get the 7 phrases with highest similarity score.
    """
    input_embeddings = model.encode([search.query], convert_to_tensor=True)
    cosine_scores = util.cos_sim(input_embeddings, all_embedding_)
    best = torch.topk(cosine_scores, 7)

    output = []
    # best[0]-> Score de los topk resultados
    # best[1]-> Índice de los topk resultados (respecto al corpus)
    for score, idx in zip(best[0][0], best[1][0]):
        score = round(float(score), 4)
        idx = int(idx)
        output.append(
            {
                "Oración": all_sentences_[idx],
                "Párrafo": all_paragraphs_[idx],
                "Score": score,
                "Documento": all_documents_[idx],
            },
        )

    out = {}
    out["resultados"] = output
    return JSONResponse(content=out)


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
