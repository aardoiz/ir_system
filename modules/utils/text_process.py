import re
from typing import List

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import spacy

nlp = spacy.load('es_core_news_sm')

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("omw-1.4")


stop_words = set(stopwords.words("spanish"))
snowball = SnowballStemmer("spanish")


def preprocess(text: str) -> str:
    """
    For any text, this function does the following:
        Replace complex expression for unicode chars < >.
        Take out puntuaction and digits.
        Convert all the character to lowercase.
        Take out stopwords and get the stemma for each word. Faster than use spacy to get the lemmas.
    Lastly, returns the processed text
    """

    text = re.sub("&lt;/?", "<", text)
    text = re.sub("&gt;", ">", text)
    text = re.sub("(\d|\.|,)+", " ", text)
    text = text.lower()

    text = text.split()
    text = " ".join([snowball.stem(word) for word in text if not word in stop_words])

    return text

def e_preprocess(text: str) -> str:
    """
    For any text, this function does the following:
        Replace complex expression for unicode chars < >.
        Take out puntuaction and digits.
        Convert all the character to lowercase.
        Take out stopwords and get the stemma for each word. Faster than use spacy to get the lemmas.
    Lastly, returns the processed text
    """

    # Conversión al lema
    doc = nlp(text)
    text = ' '.join(t.lemma_ for t in doc)  # ---> Spacy
    #text = " ".join([snowball.stem(word) for word in text.split(' ')])  # ---> NLTK

    # Elimiación de los caracteres no alfanumericos
    text = re.sub("&lt;/?", "<", text) 
    text = re.sub("&gt;", ">", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"[^a-zA-Z ,.;:áéíóúãñüÁÉÍçÇÑÓÚ\d]", "", text)

    # Paso a minúsculas
    text = text.lower()

    # Eliminación de stopwords
    text = ' '.join([word for word in text.split(' ') if word not in stop_words])


    return text

def cross_preprocess(text: str) -> str:
    """
    For any text, this function does the following:
        Replace complex expression for unicode chars < >.
        Take out puntuaction and digits.
        Convert all the character to lowercase.
        Take out stopwords and get the stemma for each word. Faster than use spacy to get the lemmas.
    Lastly, returns the processed text
    """

    # Elimiación de los caracteres no alfanumericos
    text = re.sub("&lt;/?", "<", text) 
    text = re.sub("&gt;", ">", text)
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"[^a-zA-Z ,.;:áéíóúãñüÁÉÍçÇÑÓÚ\d]", "", text)

    # Paso a minúsculas
    text = text.lower()

    return text


def mark_tag_for_html(text: str, regex: str) -> str:
    """
    Function that given a certain text and a regex, search for the regex in the text and replaces it with HTML marks to use in the front.
    """
    textos = re.findall(regex, text)
    for group in set(textos):
        text = text.replace(group, f"<mark>{group}</mark>")
    return text


def get_regex(tokenized_query: List[str]) -> str:
    """
    Function that given a list of words, creates a regex that allows to capture the words inside the list even if they appear in any case (lower or upper).
    """
    regex = ""
    for i, tok in enumerate(tokenized_query):
        r_tok = ""
        for letra in tok:
            r_tok += f"[{letra}{letra.swapcase()}]"
        if i != len(tokenized_query) - 1:
            r_tok = f"{r_tok}\S*|"
        else:
            r_tok = f"{r_tok}\S*"
        regex += r_tok
    return regex
