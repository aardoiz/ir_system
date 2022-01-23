import pickle
import re
from os import listdir

import es_core_news_sm
from environs import Env
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from sentence_transformers import SentenceTransformer

from models.document import Document

env = Env()

sentence_transformers_model = env.str(
    "SENTENCE_TRANSFORMERS_MODEL",
    "eduardofv/stsb-m-mt-es-distiluse-base-multilingual-cased-v1",
)

model = SentenceTransformer(sentence_transformers_model)
nlp = es_core_news_sm.load()


document_list = []

def process_pdfs(path):

    asignatura = path[10:]

    folder = listdir(path)


    for pdf in folder:
        tema = pdf[:-4]
        print(tema)
    # TODO:
        


        fp = open(f'{path}/{pdf}', "rb")
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        pages = PDFPage.get_pages(fp)

        # Procesamiento de un documento

        tuplas = {}
        resultado = []
        for i, page in enumerate(pages):

            # TODO: Permitir al usuario elegir que página saltarse
            if i == 1:
                continue

            tuplas[i] = []
            # print('Processing next page...')
            interpreter.process_page(page)
            layout = device.get_result()
            for lobj in layout:
                if isinstance(lobj, LTTextBox):
                    x, y, text = lobj.bbox[0], lobj.bbox[3], lobj.get_text()

                    if y > 70:
                        text = re.sub("_{2,}.+(?:\n+.+)+", "", text)
                        tuplas[i].append((y, text))
                        # print('At %r is text: %s' % ((x, y), text))
                        # print(text)

            prueba = []
            for a in tuplas[i]:
                prueba.append(a)
            prueba.sort(key=lambda y: (-y[0]))

            pagina = []
            for a in prueba:

                pagina.append(" ".join(a[1].split()))
            # print(pagina)
            resultado.append(pagina)


        # Post-procesado
        no_blanks = []
        for pagina in resultado:
            for parrafo in pagina:
                if len(parrafo) > 1:
                    clean = re.sub("•", "", parrafo).strip()
                    no_blanks.append(clean)



        b = None
        bien = []
        for i,a in enumerate(no_blanks):
            if i == 0 :
                b = a
                continue
            
                
            nums = re.sub('([\d%,\.])+','',a).strip()
            if len(nums) < 1:
                continue
                
            if b[-1].islower() or len(b) == 1:
                b = f'{b} {a}'

            if not a[-1].islower():
                if a in b:
                    bien.append(b.strip())
                else:
                    bien.append(b.strip())
                    bien.append(a.strip())
                b = ' '


        

        for element in bien:

            doc = nlp(element)
            for sent in doc.sents:

                embeddings = model.encode(sent.text, convert_to_tensor=True)

                a = Document(
                    type=asignatura,
                    document=tema,
                    paragraph=element,
                    sentence=sent.text,
                    embedding=embeddings,
                )
                document_list.append(a)
    

"""process_pdfs('data/pdfs/edicion')

with open("data/pickle/document_list.pkl", "wb") as f:
    pickle.dump(document_list, f)"""