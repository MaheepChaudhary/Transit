from imports import *
from models import *
from dataset import *


if __name__ == "__main__":
    
    model = LanguageModel("EleutherAI/pythia-70m", device_map="mps")
    
    train_data, val_data = inspect_data(data)
    
    try:
        with open("data/train_data.pkl", "rb") as f:
            train_data = pickle.load(f)
        
        with open("data/val_data.pkl", "rb") as f:
            val_dataloader = pickle.load(f)

    except:
        train_dataloader, val_dataloader = process_data(model, train_data, val_data)
        
        with open("data/train_data.pkl", "wb") as f:
            pickle.dump(train_data, f)
        
        with open("data/val_data.pkl", "wb") as f:
            pickle.dump(val_dataloader, f)
    
    print(model)
    
    with t.no_grad():
        for batch in val_dataloader:
            print(batch["input_ids"].shape) # 
            # with model.trace(batch["input_ids"]) as tracer:
            #     pass