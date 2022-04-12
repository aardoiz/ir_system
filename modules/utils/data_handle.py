import pickle

def load_local_data():
    with open('/data/pickle/document_list.pkl', 'rb') as f:
        data = pickle.load(f)
        return data

def load_sample_data():
    with open('/data/pickle/datos_sqac.pkl', 'rb') as f:
        data = pickle.load(f)
        return data