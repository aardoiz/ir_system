import argparse

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from torch import cuda, device

from modules.local_db import get_local_data
from modules.mongo_db import get_data_from_db
from modules.utils.text_process import get_regex, mark_tag_for_html, preprocess

# ArgParse Settings
parser = argparse.ArgumentParser()
parser.add_argument(
    "--database", metavar="database", type=str, help="Write Mongo if you want to use it"
)
args = parser.parse_args()
db = args.database


class Search(BaseModel):
    query: str


# FastAPI Settings
app = FastAPI()
CORS_ORIGINS = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GPU Settings
device = device("cuda" if cuda.is_available() else "cpu")
print(f"Device selected: {device}")


# DataBase Settings
if db == "Mongo":
    (
        all_documents_,
        all_titles_,
        all_sentences_,
    ) = get_data_from_db()
else:
    all_documents_, all_titles_, all_sentences_, _ = get_local_data()

# CrossEncoder Settings
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-2-v2"
cross = CrossEncoder(cross_encoder_model)

# Okapi BM25 Settings
tokenized_corpus_sentence = [preprocess(doc).split(" ") for doc in all_sentences_]
bm25_sentence = BM25Okapi(tokenized_corpus_sentence)

# Main methods
@app.get("/app")
def root():
    """
    Main method of the program, call the front service and allows user to interact using it.
    """
    return FileResponse("./front/main.html")


@app.get("/favicon.ico")
async def favicon():
    """
    Returns an icon file to use as page favicon.
    """
    return FileResponse("./front/favicon.ico")


@app.post("/bm25")
def compute_bm(search: Search) -> JSONResponse:
    """
    This method does the following steps:
    - Process the query to get all keywords
    - Get a regex to mark the query in the front
    - Use BM25 to compute the score between the query and each document of the database
    - Returns the top-7 most similar documents in JSON format
    """

    tokenized_query = preprocess(search.query).split(" ")
    regex = get_regex(tokenized_query)

    doc_scores = torch.tensor(np.array(bm25_sentence.get_scores(tokenized_query)))
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
                "Título": all_titles_[idx],
                "Score": score,
                "Documento": all_documents_[idx],
                "Oración_HTML": mark_tag_for_html(all_sentences_[idx], regex),
                "Título_HTML": mark_tag_for_html(all_titles_[idx], regex),
            }
        )

    out = {}
    out["Resultados"] = output
    return JSONResponse(content=out)


@app.post("/cross_encoder")
def compute_crossencoder(search: Search) -> JSONResponse:
    """
    This method does the following steps:
    - Process the query to get all keywords
    - Get a regex to mark the query in the front
    - Use BM25 to compute the score between the query and each document of the database
    - Returns the top-7 most similar documents
    - Create a list of strings concatenating the query and each one of the BM25 results.
    - Compute the cross-encoder score for each concatenated string
    - Re-arrange the order in base of the highest new scores
    - Return the top-5 most similar documents in JSON format
    """

    tokenized_query = preprocess(search.query).split(" ")
    regex = get_regex(tokenized_query)

    doc_scores = torch.tensor(np.array(bm25_sentence.get_scores(tokenized_query)))
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
                "Título": all_titles_[idx],
                "Score": score,
                "Documento": all_documents_[idx],
                "Oración_HTML": mark_tag_for_html(all_sentences_[idx], regex),
                "Título_HTML": mark_tag_for_html(all_titles_[idx], regex),
            },
        )

    combinations = [[search.query, sen["Oración"]] for sen in output]
    if len(combinations) > 0:
        sim_score = cross.predict(combinations)
        sim_score_argsort = reversed(np.argsort(sim_score))

        count = 0
        real_out = []
        for idx in sim_score_argsort:
            count += 1
            real_out.append(
                {
                    "Oración": output[idx]["Oración"],
                    "Título": output[idx]["Título"],
                    "Score": round(float(sim_score[idx]), 2),
                    "Documento": output[idx]["Documento"],
                    "Oración_HTML": mark_tag_for_html(output[idx]["Oración"], regex),
                    "Título_HTML": mark_tag_for_html(output[idx]["Título"], regex),
                }
            )
            if count == 5:
                break

        out = {}
        out["Resultados"] = real_out
    else:
        out = {"Resultados": []}
    return JSONResponse(content=out)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8425)
