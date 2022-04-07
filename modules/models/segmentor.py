import re
from typing import List

import es_core_news_sm
from modules.models.document import Document


nlp = es_core_news_sm.load()



def cleaner(text: str) -> List[str]:
    """
    Eliminate script tag and its content.
    Eliminate "<" and ">" markers and its content.
    Eliminate \t.

    Subdivide the text into paragraphs.

    """
    body = re.sub("<script>(?:.+\n|\n)*<\/script>", "", text)
    body = re.sub("<.+?>", "", body)
    body = re.sub("\t", " ", body)

    paragraph_list = [n for n in body.split("\n\n") if len(n) > 0]

    new_para = []
    for element in paragraph_list:
        processed = " ".join([e for e in element.split()]).strip()
        if len(processed) > 0:
            new_para.append(processed)

    return new_para
