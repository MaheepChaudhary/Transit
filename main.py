from imports import *
from models import *

model = LanguageModel("EleutherAI/pythia-70m", device_map="mps")

for d in data:
    print(d)
    break