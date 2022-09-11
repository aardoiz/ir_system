import pickle
from sentence_transformers import CrossEncoder
import numpy as np
from torch import cuda, device
from time import time


device = device("cuda" if cuda.is_available() else "cpu")
print(f"Device selected: {device}")

cross_encoder_model = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
cross = CrossEncoder(cross_encoder_model, device=device)

with open('datos_sqac.pkl', 'rb') as f:
    data = pickle.load(f)

search = 'La época dorada del imperio británico'

combinations = [[search, sen["content"]] for sen in data]

out = []
for i in range(5):
  init = time()
  sim_score = cross.predict(combinations)
  sim_score_argsort = reversed(np.argsort(sim_score))
  out.append(time()-init)

with open('out.txt','w') as f:
    for iteration in out:
        f.write(f'{iteration}\n')