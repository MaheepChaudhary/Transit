from imports import *
from models import *
from dataset import *

def activation_embeds(model, train_dataloader):
    
    activation_embeds = []
    
    with t.no_grad():
        
        for batch in tqdm(train_dataloader):
            
            with model.trace(batch["input_ids"]) as tracer:
                output = model.embed_out.output.save()
            
            activation_embeds.append(output)
            
    return activation_embeds

def visualization(mean_acts):
    
    plt.figure(figsize=(10, 10))
    plt.imshow(mean_acts, cmap="viridis")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    
    model = LanguageModel("EleutherAI/pythia-70m", device_map=t.device("cuda" if t.cuda.is_available() else "mps"))
    
    train_data, val_data = inspect_data(data)
    
    try:
        with open("data/train_data.pkl", "rb") as f:
            train_dataloader = pickle.load(f)
        
        with open("data/val_data.pkl", "rb") as f:
            val_dataloader = pickle.load(f)

    except:
        train_dataloader, val_dataloader = process_data(model, train_data, val_data)
        
        with open("data/train_data.pkl", "wb") as f:
            pickle.dump(train_dataloader, f)
        
        with open("data/val_data.pkl", "wb") as f:
            pickle.dump(val_dataloader, f)
    
    train_activation_embeds = activation_embeds(model, train_dataloader)
    assert train_activation_embeds[0].shape == (8, 128, 768)
    mean_acts = np.array(train_activation_embeds).mean(axis=0)
    
    visualization(mean_acts)