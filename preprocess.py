#libraries for text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('wordnet')
nltk.download('stopwords')

def preprocess(text):
    """
    Takes out stopwords
    Takes out special char
    Takes out puntuaction and digits
    Pone to en minúsculas
    """
    #define stopwords
    stop_words = set(stopwords.words("spanish")) 
    #Remove punctuations
    text = re.sub('[^a-zA-ZíóúáéñÑÁÉÍÓÚü]', ' ', text)
    #Convert to lowercase
    text = text.lower()
    #remove tags
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    # remove special characters and digits
    text=re.sub("(\\d|\\W)+"," ",text) # CAMBIAR A \d \W
    ##Convert to list from string
    text = text.split()
    ##Stemming
    snowball = SnowballStemmer('spanish')
    text = [snowball.stem(word) for word in text if not word in stop_words]
    #Lemmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  stop_words] 
    text = " ".join(text) 
    
    return text
