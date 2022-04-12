import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from typing import List

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("omw-1.4")


stop_words = set(stopwords.words("spanish"))
snowball = SnowballStemmer("spanish")


def Preprocess(text: str) -> str:
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


def eval_preproces(text: str) -> str:
    """
    Special preprocess used in the evaluation phase of the project
    """
    text = re.sub(r"\\n", " ", text)
    text = re.sub(r"[><]", " ", text)
    text = re.sub(r"[^a-zA-Z ,.;:áéíóúãñüÁÉÍçÇÑÓÚ\d]", "", text)

    return text


def html_mark(text: str, regex: str) -> str:
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
