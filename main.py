from imports import *
from models import *
from dataset import *


if __name__ == "__main__":
    
    model = LanguageModel("EleutherAI/pythia-70m", device_map="mps")
    
    train_data, val_data = inspect_data()
    
    
    try:
        # with open("data/train_data.pkl", "rb") as f:
        #     train_data = pickle.load(f)
        
        with open("data/val_data.pkl", "rb") as f:
            val_data = pickle.load(f)

    except:
        # train_dataloader, val_dataloader = process_data(model, train_data, val_data)
        val_data = process_data(model, train_data, val_data)
        # with open("data/train_data.pkl", "wb") as f:
        #     pickle.dump(train_data, f)
        
        with open("data/val_data.pkl", "wb") as f:
            pickle.dump(val_data, f)
    
    print(model)
    for batch in list(val_data):
        print(batch)
    # with t.no_grad():
    #     for batch in val_data:
    #         with model.trace(batch["input_ids"]) as tracer:
    #             pass