from imports import *

pythia14m = LanguageModel("EleutherAI/pythia-14m", device_map="cuda")
pythia70m = LanguageModel("EleutherAI/pythia-70m", device_map="cuda")
pythia160m = LanguageModel("EleutherAI/pythia-160m", device_map="cuda")
pythia410m = LanguageModel("EleutherAI/pythia-410m", device_map="cuda")
pythia1b = LanguageModel("EleutherAI/pythia-1b", device_map="cuda")
pythia1_4b = LanguageModel("EleutherAI/pythia-1.4b", device_map="cuda")

print("Printing pythia 14m model")
print(pythia14m)

print()

print("Printing pythia 70m model")
print(pythia70m)

print()

print("Printing pythia 160m model")
print(pythia160m)

print()

print("Printing pythia 410m model")
print(pythia410m)

print()

print("Printing pythia 1b model")

print(pythia1b)
print()

print("Printing pythia 1.4b model")
print(pythia1_4b)
print()