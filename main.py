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
                output0 = model.gpt_neox.layers[0].mlp.output[0].save()
                output1 = model.gpt_neox.layers[1].mlp.output[0].save()
                output2 = model.gpt_neox.layers[2].mlp.output[0].save()
                output3 = model.gpt_neox.layers[3].mlp.output[0].save()
                output4 = model.gpt_neox.layers[4].mlp.output[0].save()
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
            
    with open("data/activation_embeds_prenorm.pkl", "wb") as f:
        pickle.dump(activation_embeds, f)
            
    return activation_embeds

def norm(activation_embeds):
    
    activation_embeds["layer 0"] = np.linalg.norm(activation_embeds["layer 0"], axis=0)/10995
    activation_embeds["layer 1"] = np.linalg.norm(activation_embeds["layer 1"], axis=0)/10995
    activation_embeds["layer 2"] = np.linalg.norm(activation_embeds["layer 2"], axis=0)/10995
    activation_embeds["layer 3"] = np.linalg.norm(activation_embeds["layer 3"], axis=0)/10995
    activation_embeds["layer 4"] = np.linalg.norm(activation_embeds["layer 4"], axis=0)/10995
    activation_embeds["last layer"] = np.linalg.norm(activation_embeds["last layer"], axis=0)/10995

    # Additional norm calculations for nested structures
    activation_embeds["layer 0"] = np.linalg.norm(activation_embeds["layer 0"], axis=1)
    activation_embeds["layer 1"] = np.linalg.norm(activation_embeds["layer 1"], axis=1)
    activation_embeds["layer 2"] = np.linalg.norm(activation_embeds["layer 2"], axis=1)
    activation_embeds["layer 3"] = np.linalg.norm(activation_embeds["layer 3"], axis=1)
    activation_embeds["layer 4"] = np.linalg.norm(activation_embeds["layer 4"], axis=1)
    activation_embeds["last layer"] = np.linalg.norm(activation_embeds["last layer"], axis=1)
    
    assert activation_embeds["layer 0"].shape == activation_embeds["layer 1"].shape == activation_embeds["layer 2"].shape == activation_embeds["layer 3"].shape == activation_embeds["layer 4"].shape == (128,)
    assert activation_embeds["last layer"].shape == (128,)
    
    actlist = np.array([
        activation_embeds["layer 0"], 
        activation_embeds["layer 1"], 
        activation_embeds["layer 2"], 
        activation_embeds["layer 3"], 
        activation_embeds["layer 4"], 
        # mean_acts["last layer"]
        ])
    
    print(actlist)

    # Create the heatmap
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.imshow(actlist, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

    # Add color bar to indicate the scale
    plt.colorbar()

    # Set labels
    plt.xlabel('Dimensions')
    plt.ylabel('Rows')

    # Optionally, you can add titles
    plt.title('Heatmap of 128-Dimension Data for 5 Rows')

    # Show the heatmap
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
        with open("data/activation_embeds_prenorm.pkl", "rb") as f:
            activation_embeds = pickle.load(f)
    except:
        activation_embeds = activation_embeds_fn(model, val_dataloader)
        with open("data/activation_embeds_prenorm.pkl", "wb") as f:
            pickle.dump(activation_embeds, f)
    
    # For batch size 2 so we will take the mean and then will divide by len(val_dataloader)

    # Replace mean with norm
    print(activation_embeds["layer 0"].shape)
    
    
    norm(activation_embeds)
    normwmean(activation_embeds)