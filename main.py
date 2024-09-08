from imports import *
from models import *
from dataset import *

def create_dataloader(tokenized_data, batch_size=2):
    input_ids = torch.tensor([item['input_ids'] for item in tokenized_data])
    attention_mask = torch.tensor([item['attention_mask'] for item in tokenized_data])

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# Create DataLoader


def activation_embeds_fn(model, dataloader): # So it contains 5 layers and one last layer. 
    model.eval()
    
    activation_embeds = {}
    activation_embeds["layer 0"] = activation_embeds["layer 1"] = activation_embeds["layer 2"] = activation_embeds["layer 3"] = activation_embeds["layer 4"] = t.zeros((args.batch_size, 128, 512))
    activation_embeds["last layer"] = t.zeros((args.batch_size, 128, 50304))
    
    with t.no_grad():
        for batch in tqdm(val_dataloader):
            
            with model.trace(batch["input_ids"]) as tracer:
                output0 = model.gpt_neox.layers[0].output[0].save()
                output1 = model.gpt_neox.layers[1].output[0].save()
                output2 = model.gpt_neox.layers[2].output[0].save()
                output3 = model.gpt_neox.layers[3].output[0].save()
                output4 = model.gpt_neox.layers[4].output[0].save()
                output = model.embed_out.output.save()

            activation_embeds["layer 0"]+=output0
            activation_embeds["layer 1"]+=output1
            activation_embeds["layer 2"]+=output2
            activation_embeds["layer 3"]+=output3
            activation_embeds["layer 4"]+=output4
            activation_embeds["last layer"]+=output
            

            del batch
            del output
            gc.collect()
            
    with open("data/activation_embeds.pkl", "wb") as f:
        pickle.dump(activation_embeds, f)
            
    return activation_embeds

def visualization(mean_acts):
    
    actlist = [mean_acts["layer 0"], mean_acts["layer 1"], mean_acts["layer 2"], mean_acts["layer 3"], mean_acts["layer 4"], mean_acts["last layer"]]
    
    print(actlist)
    plt.plot(actlist)
    plt.title('Activation Embeddings Across Layers')
    plt.xlabel('Layers')
    plt.ylabel('Activation Value (Averaged)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default="8")
    
    args = parser.parse_args()
    
    
    # model = LanguageModel("EleutherAI/pythia-70m", device_map=t.device("cuda" if t.cuda.is_available() else "mps"))
    model = LanguageModel("EleutherAI/pythia-70m", device_map="cpu")
    print(model)
    train_data, val_data = inspect_data(data)
    
    try:
        # with open("data/train_data.pkl", "rb") as f:
        #     train_dataloader = pickle.load(f)
        
        with open(f"data/val_data_b{args.batch_size}.pkl", "rb") as f:
            val_dataloader = pickle.load(f)

    except:
        val_dataloader = process_data(model, train_data, val_data, args.batch_size)
        
        # with open("data/train_data.pkl", "wb") as f:
        #     pickle.dump(train_dataloader, f)
        
        with open(f"data/val_data_b{args.batch_size}.pkl", "wb") as f:
            pickle.dump(val_dataloader, f)
    
    try:
        with open("data/activation_embeds.pkl", "rb") as f:
            activation_embeds = pickle.load(f)
    except:
        activation_embeds = activation_embeds_fn(model, val_dataloader)
        with open("data/activation_embeds.pkl", "wb") as f:
            pickle.dump(activation_embeds, f)
    
    # For batch size 2 so we will take the mean and then will divide by len(val_dataloader)
    activation_embeds["layer 0"] = np.array(activation_embeds["layer 0"]).mean(axis=0)/10995
    activation_embeds["layer 1"] = np.array(activation_embeds["layer 1"]).mean(axis=0)/10995
    activation_embeds["layer 2"] = np.array(activation_embeds["layer 2"]).mean(axis=0)/10995
    activation_embeds["layer 3"] = np.array(activation_embeds["layer 3"]).mean(axis=0)/10995
    activation_embeds["layer 4"] = np.array(activation_embeds["layer 4"]).mean(axis=0)/10995
    activation_embeds["last layer"] = np.array(activation_embeds["last layer"]).mean(axis=0)/10995
    
    activation_embeds["layer 0"] = activation_embeds["layer 0"].mean(axis = 0).mean(axis = 0)
    activation_embeds["layer 1"] = activation_embeds["layer 1"].mean(axis = 0).mean(axis = 0)
    activation_embeds["layer 2"] = activation_embeds["layer 2"].mean(axis = 0).mean(axis = 0)
    activation_embeds["layer 3"] = activation_embeds["layer 3"].mean(axis = 0).mean(axis = 0)
    activation_embeds["layer 4"] = activation_embeds["layer 4"].mean(axis = 0).mean(axis = 0)
    activation_embeds["last layer"] = activation_embeds["last layer"].mean(axis = 0).mean(axis = 0)
    
    visualization(activation_embeds)