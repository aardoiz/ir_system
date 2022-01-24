# libraries for text preprocessing
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download("wordnet")
nltk.download("stopwords")
nltk.download("omw-1.4")


stop_words = set(stopwords.words("spanish"))
snowball = SnowballStemmer("spanish")

def Preprocess(text:str) -> str:
    """
    For any text, this function does the following:
        Replace complex expression for unicode chars < >.
        Take out puntuaction and digits.
        Convert all the character to lowercase.
        Take out stopwords and get the stemma for each word.
    """

    
    text = re.sub("&lt;/?", "<", text)
    text = re.sub("&gt;", ">", text)
    text = re.sub("(\d|\.|,)+", " ", text)
    text = text.lower()

    text = text.split()
    text = " ".join([snowball.stem(word) for word in text if not word in stop_words])

    return text
