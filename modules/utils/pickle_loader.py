import pickle

def load_local():
    # Load the local db pickle file
    with open("./data/pickle/my_data.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def load_sample():
    # Load the sample pickle file
    with open("./data/pickle/sample.pkl", "rb") as f:
        data = pickle.load(f)
    return data