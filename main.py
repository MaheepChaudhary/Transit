from imports import *
from models import *
from dataset import *


if __name__ == "__main__":
    model = LanguageModel("EleutherAI/pythia-70m", device_map="mps")
    train_data, val_data = inspect_data()
    train_dataloader, val_dataloader = process_data(model, train_data, val_data)