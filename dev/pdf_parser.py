from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from os import listdir

import re
import es_core_news_sm

nlp = es_core_news_sm.load()


def carga_docs(carpeta):
    # TODO:
    pass


asignatura = "edicion"
tema = "Unidad8_Industria"

doc = f"data/pdfs/{asignatura}/{tema}.pdf"

js = {}
js["name"] = f"{doc[:-4]}.json"  # Apañar

# Operaciones pdfminer

fp = open(doc, "rb")
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

print(no_blanks)
with open(f"data/txt/{asignatura}/{tema}.txt", "w") as f:
    for line in no_blanks:
        f.write(line)
