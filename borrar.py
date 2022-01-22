import pickle
import torch
from environs import Env


def read_pickle(pickle_file: str, object_type: str):
    """
    Read a .pkl file and loads into memory.
    It's used for acessing the corpus of the IR system
    """
    with open(pickle_file, "rb") as fich:
        if object_type == "tensor":
            read_file = torch.load(fich, map_location="cpu")
        elif object_type == "list":
            read_file = pickle.load(fich)
        else:
            raise ("Error: No available object type")
    return read_file

env = Env()
super_path = "data/"
embeddings = env.str("EMBEDDINGS", f"{super_path}pickle/embeddings.pt")
all_embedding_from_source = read_pickle(embeddings, "tensor")

print(type(all_embedding_from_source[0]))
print(all_embedding_from_source[0])