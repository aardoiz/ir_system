import pickle
import re
from os import listdir

from modules.models.segmentor import sentencizer
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage


def process_pdfs(path: str):
    """
    First, check if there is a local db stored. 
    Then, given a folder path, get the files inside it and do the following for each pdf file:
        - Get all data information that is outside tables and images.
        - Process all strings and deletes errors from the parser.
        - Divide the data into sentences and do the following:
            - Calculate its embeddings using SBERT model
            - Store the information in a Document object which includes subject, lesson, paragraph, sentence and embeddings
    At the end, store all documents information into a pickle object to use it on the next step.
    """
    boolie = False
    if len(listdir("data/pickle")) > 1:
        with open('data/pickle/document_list.pkl', 'rb') as f:
            data = pickle.load(f)
            boolie = True

    document_list = []

    folder = listdir(path)

    for pdf in folder:

        if pdf[-4:] != ".pdf":
            continue
        print(pdf)

        fp = open(f"{path}/{pdf}", "rb")
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
            #print(pagina)
            resultado.append(pagina)
            no_blanks = []
        for pagina in resultado:
            nano = []
            for parrafo in pagina:
                if len(parrafo) > 1:
                    clean = re.sub("•", "", parrafo).strip()
                    nano.append(clean)
            no_blanks.append(nano)
            
        for a in no_blanks:
            texto = ' '.join(a)
            #print(texto)
            #print('----')

            document_list.append({"title":pdf, "content":texto})

    if boolie:
        data.extend(document_list)
    else:
        data = document_list
    # Create the pkl to use in next step
    with open("data/pickle/document_list.pkl", "wb") as f:
        pickle.dump(data, f)

process_pdfs('data/pdfs/edicion')