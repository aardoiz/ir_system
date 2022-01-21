import requests
from bs4 import BeautifulSoup
import es_core_news_sm

nlp = es_core_news_sm.load()
text = "mi perro es super de Amazon"

doc = nlp(text)
for t in doc:
    print(t.lemma_)
"""r = requests.get('https://ariesco.github.io/OIM/docs/intro.html')
soup = BeautifulSoup(r.text, 'lxml')"""
