from pdfminer.layout import LAParams, LTTextBox
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
import json
import time 
from os import listdir

import re


# Carga de docs
def procesador_pdf():
    documents = listdir('../data/pdfs/edicion/')
    for doc in documents:
        
        js = {}
        js['name'] = f'{doc[:-4]}.json' 


        fp = open(f'../data/pdfs/edicion/{doc}', 'rb')
        rsrcmgr = PDFResourceManager()
        laparams = LAParams()
        device = PDFPageAggregator(rsrcmgr, laparams=laparams)
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        pages = PDFPage.get_pages(fp)

        tuplas = {}
        frases_final = []

        lista_pruebas_solo_jupyter = []
        lista_buena = []
        lista_final = []
        for i,page in enumerate(pages):
    
            if i == 1: # SALTAR LA PÁGINA 2 (ÍNDICE)
                continue
        
            tuplas[i] = []
            # print('Processing next page...')
            interpreter.process_page(page)
            layout = device.get_result()
            for lobj in layout:
                if isinstance(lobj, LTTextBox):
                    x, y, text = lobj.bbox[0], lobj.bbox[3], lobj.get_text()

                    if y > 70:
                        text = re.sub('_{2,}.+(?:\n+.+)+','',text)
                        tuplas[i].append((y, text))
                        #print('At %r is text: %s' % ((x, y), text))
                        #print(text)
                        
            prueba = []
            for a in tuplas[i]:
                prueba.append(a)
            prueba.sort(key=lambda y: ( -y[0]))
            
            pagina = []
            for a in prueba:
                
                pagina.append(' '.join(a[1].split()))
            # print(pagina)
        
procesador_pdf()