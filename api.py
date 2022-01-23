import pickle

import numpy as np
import torch
import uvicorn
from environs import Env
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer, util
from torch import cuda, device

from preprocess import Preprocess


class Search(BaseModel):
    query: str


def read_pickle(pickle_file: str, object_type: str):
    """
    Read a .pkl file and loads into memory.
    It's used for acessing the corpus of the IR system
    """
    with open(pickle_file, "rb") as fich:
        if object_type == "tensor":
            read_file = torch.load(fich, map_location="cpu")
        elif object_type == "list":
            read_file = pickle.load(fich)
        else:
            raise ("Error: No available object type")
    return read_file


# Environs
env = Env()
env.read_env()

super_path = "data/"

embeddings = env.str("EMBEDDINGS", f"{super_path}pickle/embeddings.pt")
sentences = env.str("SENTENCES", f"{super_path}pickle/sentences.pkl")
paragraphs = env.str("PARAGRAPHS", f"{super_path}pickle/paragraphs.pkl")
documents = env.str("DOCUMENTS", f"{super_path}pickle/documents.pkl")

sentence_transformers_model = env.str(
    "SENTENCE_TRANSFORMERS_MODEL",
    "eduardofv/stsb-m-mt-es-distiluse-base-multilingual-cased-v1",
)

cross_encoder_model = env.str(
    "CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2"
)

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
all_embedding_from_source = read_pickle(embeddings, "tensor")
all_sentences_from_source = read_pickle(sentences, "list")
all_documents_from_source = read_pickle(documents, "list")
all_paragraphs_from_source = read_pickle(paragraphs, "list")

model = SentenceTransformer(sentence_transformers_model)
cross = CrossEncoder(cross_encoder_model)

# Okapi BM25
tokenized_corpus_sentence = [
    Preprocess(doc).split(" ") for doc in all_sentences_from_source
]
tokenized_corpus_paragraph = [
    Preprocess(doc).split(" ") for doc in all_paragraphs_from_source
]

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
    - Calculates the cosine distance between this embedding and all embeddings pre-computed from the corpus.
    - Get the 7 phrases which similarity scored highest.
    """
    input_embeddings = model.encode([search.query], convert_to_tensor=True)
    cosine_scores = util.pytorch_cos_sim(input_embeddings, all_embedding_from_source)
    best = torch.topk(cosine_scores, 7)

    output = []
    # best[0]-> Score de los topk resultados
    # best[1]-> Índice de los topk resultados (respecto al corpus)
    for score, idx in zip(best[0][0], best[1][0]):
        score = round(float(score), 4)
        idx = int(idx)
        output.append(
            {
                "Oración": all_sentences_from_source[idx],
                "Párrafo": all_paragraphs_from_source[idx],
                "Score": score,
                "Documento": all_documents_from_source[idx],
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
    - Use BM25 to compute the socre between the query and each sentence.
    - Returns the 7 results with highest score.
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
                "Oración": all_sentences_from_source[idx],
                "Párrafo": all_paragraphs_from_source[idx],
                "Score": score,
                "Documento": all_documents_from_source[idx],
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
    - Use BM25 to compute the socre between the query and each sentence.
    - Returns the 7 results with highest score.
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
                "Oración": all_sentences_from_source[idx],
                "Párrafo": all_paragraphs_from_source[idx],
                "Score": score,
                "Documento": all_documents_from_source[idx],
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
