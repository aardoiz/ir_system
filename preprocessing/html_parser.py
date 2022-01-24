import pickle

import requests
from bs4 import BeautifulSoup
from tfm.preprocessing.models.segmentor import cleaner, sentencizer

# TODO: Selector múltiple/lector de json

web_url = "https://ariesco.github.io/OIM/docs/intro.html"
asignatura = "OIM"
tema = "Introducción"


r = requests.get("https://ariesco.github.io/OIM/docs/intro.html")
soup = BeautifulSoup(r.text, "html.parser")
body = str(soup.body)

paragraphs = cleaner(body)
document_list = sentencizer(paragraphs, subject=asignatura, document=tema)


with open("data/pickle/document_list.pkl", "wb") as f:
    pickle.dump(document_list, f)
