import re
import es_core_news_sm
from models.document import Document
from typing import List
from sentence_transformers import SentenceTransformer


nlp = es_core_news_sm.load()
model = SentenceTransformer("eduardofv/stsb-m-mt-es-distiluse-base-multilingual-cased-v1")



def cleaner(text:str)-> List[str]:
    """
    Eliminate script tag and its content.
    Eliminate "<" and ">" markers and its content.
    Eliminate \t.

    Subdivide the text into paragraphs.

    """
    body = re.sub('<script>(?:.+\n|\n)*<\/script>','',text)
    body = re.sub('<.+?>','', body)
    body = re.sub('\t', ' ', body)

    paragraph_list = [n for n in body.split('\n\n') if len(n) >0]

    new_para = []
    for element in paragraph_list:  
        processed = ' '.join([e for e in element.split()]).strip()
        if len(processed) > 0:
            new_para.append(processed)

    return new_para


def sentencizer(paragraph_list:List, subject:str, document: str) -> List[Document]:
    """
    Initialize a blank list to store Document Objects.
    For each paragraph of the input list, divide it into sentences.
    For each sentence, get the embedding using SBERT and create a Document Object."""
    
    document_list = []
    for element in paragraph_list:

            doc = nlp(element)
            for sent in doc.sents:
                embeddings = model.encode(sent.text, convert_to_tensor=True)

                a = Document(
                    type=subject,
                    document=document,
                    paragraph=element,
                    sentence=sent.text,
                    embedding=embeddings,
                )
                document_list.append(a)

    return document_list
