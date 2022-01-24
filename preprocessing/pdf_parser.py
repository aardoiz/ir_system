import pickle
import re
from os import listdir

from models.segmentor import sentencizer
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage
from sentence_transformers import SentenceTransformer

model = SentenceTransformer(
    "eduardofv/stsb-m-mt-es-distiluse-base-multilingual-cased-v1"
)


def process_pdfs(path: str):
    """
    Given a folder path, get the files inside it and do the following for each pdf file:
        - Get all data information that is outside tables and images.
        - Process all strings and deletes errors from the parser.
        - Divide the data into sentences and do the following:
            - Calculate its embeddings using SBERT model
            - Store the information in a Document object which includes subject, lesson, paragraph, sentence and embeddings
    At the end, store all documents information into a pickle object to use it on the next step.
    """

    document_list = []

    asignatura = path[10:]

    folder = listdir(path)

    for pdf in folder:

        if pdf[-4:] != ".pdf":
            continue

        tema = pdf[:-4]

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
        for i, a in enumerate(no_blanks):
            if i == 0:
                b = a
                continue

            nums = re.sub("([\d%,\.])+", "", a).strip()
            if len(nums) < 1:
                continue

            if b[-1].islower() or len(b) == 1:
                b = f"{b} {a}"

            if not a[-1].islower():
                if a in b:
                    bien.append(b.strip())
                else:
                    bien.append(b.strip())
                    bien.append(a.strip())
                b = " "

        ea = sentencizer(bien, asignatura, tema)
        document_list.extend(ea)
    print(len(document_list))
    print([(e.sentence, e.paragraph) for e in document_list[:100]])
    # Create the pkl to use in next step
    with open("data/pickle/document_list.pkl", "wb") as f:
        pickle.dump(document_list, f)


process_pdfs("data/pdfs/edicion")
