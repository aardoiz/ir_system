from pydantic import BaseModel
from torch import Tensor


class Document(BaseModel):
    type: str
    document: str
    paragraph: str
    sentence: str
    embedding: Tensor

    class Config:
        arbitrary_types_allowed = True
