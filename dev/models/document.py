from torch import Tensor
from pydantic import BaseModel

class Document(BaseModel):
    document: str
    paragraph: str 
    sentence: str
    embedding: Tensor 