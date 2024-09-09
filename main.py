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

def activation_embeds_fn(model, dataloader, batch_size): # So it contains 5 layers and one last layer. 
    model.eval()
    
    activation_embeds = {
        "layer 0": [],
        "layer 1": [],
        "layer 2": [],
        "layer 3": [],
        "layer 4": []
    }
    
    with t.no_grad():
        for batch in tqdm(dataloader):
            
            with model.trace(batch["input_ids"]) as tracer:
                output0 = model.gpt_neox.layers[0].mlp.act.output.save()
                output1 = model.gpt_neox.layers[1].mlp.act.output.save()
                output2 = model.gpt_neox.layers[2].mlp.act.output.save()
                output3 = model.gpt_neox.layers[3].mlp.act.output.save()
                output4 = model.gpt_neox.layers[4].mlp.act.output.save()
                output = model.embed_out.output.save()

            # output0.shape -> (batch_size, 128, 2048)
            activation_embeds["layer 0"].append(t.norm(t.norm(output0, dim = 0), dim = -1))
            activation_embeds["layer 1"].append(t.norm(t.norm(output1, dim = 0), dim = -1))
            activation_embeds["layer 2"].append(t.norm(t.norm(output2, dim = 0), dim = -1))
            activation_embeds["layer 3"].append(t.norm(t.norm(output3, dim = 0), dim = -1))
            activation_embeds["layer 4"].append(t.norm(t.norm(output4, dim = 0), dim = -1))
            
    with open("data/activation_embeds_prenorm.pkl", "wb") as f:
        pickle.dump(activation_embeds, f)
            
    return activation_embeds


def plotting(data, name):
    # Create the heatmap
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.imshow(data, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

    # Add color bar to indicate the scale
    plt.colorbar()

    # Set labels
    plt.xlabel('Tokens')
    plt.ylabel('Layers')

    # Optionally, you can add titles
    if name == "figures/layer_seq_norm.png" or name == "figures/layer_seq_normwmean.png":
        plt.title('Token activations in different layers')
    elif name == "figures/grads_layer_seq_norm.png" or name == "figures/grad_layer_seq_normwmean.png":
        plt.title('Token gradients in different layers')

    # Show the heatmap
    plt.savefig(name)

class normed:

    def __init__(self, actemb):
        self.actemb = actemb
        
        


        
    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        norm_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        
        norm_actemb["layer 0"] = np.linalg.norm(self.actemb["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.linalg.norm(self.actemb["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.linalg.norm(self.actemb["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.linalg.norm(self.actemb["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.linalg.norm(self.actemb["layer 4"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        print(np.array(self.actemb["layer 0"]).shape)
        
        actlist = np.array([
            np.log(np.array(norm_actemb["layer 0"])),
            np.log(np.array(norm_actemb["layer 1"])),
            np.log(np.array(norm_actemb["layer 2"])),
            np.log(np.array(norm_actemb["layer 3"])),
            np.log(np.array(norm_actemb["layer 4"])),
            # mean_acts["last layer"]
            ])
        print(actlist.shape)
        plotting(data=actlist, name = "figures/layer_seq_norm.png")


    def normwmean(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        normwmean_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        normwmean_actemb_mod = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        normwmean_actemb["layer 0"] = np.array(self.actemb["layer 0"]) - np.mean(np.array(self.actemb["layer 0"]), axis = 0)
        normwmean_actemb["layer 1"] = np.array(self.actemb["layer 1"]) - np.mean(np.array(self.actemb["layer 1"]), axis = 0)
        normwmean_actemb["layer 2"] = np.array(self.actemb["layer 2"]) - np.mean(np.array(self.actemb["layer 2"]), axis = 0)
        normwmean_actemb["layer 3"] = np.array(self.actemb["layer 3"]) - np.mean(np.array(self.actemb["layer 3"]), axis = 0)
        normwmean_actemb["layer 4"] = np.array(self.actemb["layer 4"]) - np.mean(np.array(self.actemb["layer 4"]), axis = 0)
        
        normwmean_actemb_mod["layer 0"] = np.linalg.norm(normwmean_actemb["layer 0"], axis=0)
        normwmean_actemb_mod["layer 1"] = np.linalg.norm(normwmean_actemb["layer 1"], axis=0)
        normwmean_actemb_mod["layer 2"] = np.linalg.norm(normwmean_actemb["layer 2"], axis=0)
        normwmean_actemb_mod["layer 3"] = np.linalg.norm(normwmean_actemb["layer 3"], axis=0)
        normwmean_actemb_mod["layer 4"] = np.linalg.norm(normwmean_actemb["layer 4"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        
        actlistmean = np.array([
            np.log(np.array(normwmean_actemb_mod["layer 0"])),
            np.log(np.array(normwmean_actemb_mod["layer 1"])),
            np.log(np.array(normwmean_actemb_mod["layer 2"])),
            np.log(np.array(normwmean_actemb_mod["layer 3"])),
            np.log(np.array(normwmean_actemb_mod["layer 4"])),
            # mean_acts["last layer"]
            ])

        plotting(data=actlistmean, name = "figures/layer_seq_normwmean.png")


class gradients_norm:
    
    def __init__(self, model, dataloader):
        
        self.model = model
        self.dataloader = dataloader
        
        try:
            with open("data/grads.pkl", "rb") as f:
                grads = pickle.load(f)    
            self.grads = grads  
        except:
            self.grads = self.get_grads()
            
    
    
    def get_grads(self):
        
        grad_embeds = {
        "layer 0": [],
        "layer 1": [],
        "layer 2": [],
        "layer 3": [],
        "layer 4": []
        }
        
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
            
                output0 = self.model.gpt_neox.layers[0].mlp.output.grad.save()
                output1 = self.model.gpt_neox.layers[1].mlp.output.grad.save()
                output2 = self.model.gpt_neox.layers[2].mlp.output.grad.save()
                output3 = self.model.gpt_neox.layers[3].mlp.output.grad.save()
                output4 = self.model.gpt_neox.layers[4].mlp.output.grad.save()
                
                self.model.output.logits.sum().backward()
            
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            grad_embeds["layer 0"].append(t.norm(t.norm(output0, dim = 0), dim = -1))
            grad_embeds["layer 1"].append(t.norm(t.norm(output1, dim = 0), dim = -1))
            grad_embeds["layer 2"].append(t.norm(t.norm(output2, dim = 0), dim = -1))
            grad_embeds["layer 3"].append(t.norm(t.norm(output3, dim = 0), dim = -1))
            grad_embeds["layer 4"].append(t.norm(t.norm(output4, dim = 0), dim = -1))
            
        with open("data/grads.pkl", "wb") as f:
            pickle.dump(grad_embeds, f)
            
        return grad_embeds

    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        normwmean_grad = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        normwmean_grad_mod = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        normwmean_grad["layer 0"] = np.array(self.actemb["layer 0"]) - np.mean(np.array(self.actemb["layer 0"]), axis = 0)
        normwmean_grad["layer 1"] = np.array(self.actemb["layer 1"]) - np.mean(np.array(self.actemb["layer 1"]), axis = 0)
        normwmean_grad["layer 2"] = np.array(self.actemb["layer 2"]) - np.mean(np.array(self.actemb["layer 2"]), axis = 0)
        normwmean_grad["layer 3"] = np.array(self.actemb["layer 3"]) - np.mean(np.array(self.actemb["layer 3"]), axis = 0)
        normwmean_grad["layer 4"] = np.array(self.actemb["layer 4"]) - np.mean(np.array(self.actemb["layer 4"]), axis = 0)
        
        normwmean_grad_mod["layer 0"] = np.linalg.norm(normwmean_grad["layer 0"], axis=0)
        normwmean_grad_mod["layer 1"] = np.linalg.norm(normwmean_grad["layer 1"], axis=0)
        normwmean_grad_mod["layer 2"] = np.linalg.norm(normwmean_grad["layer 2"], axis=0)
        normwmean_grad_mod["layer 3"] = np.linalg.norm(normwmean_grad["layer 3"], axis=0)
        normwmean_grad_mod["layer 4"] = np.linalg.norm(normwmean_grad["layer 4"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        
        gradlistmean = np.array([
            np.log(np.array(normwmean_grad_mod["layer 0"])),
            np.log(np.array(normwmean_grad_mod["layer 1"])),
            np.log(np.array(normwmean_grad_mod["layer 2"])),
            np.log(np.array(normwmean_grad_mod["layer 3"])),
            np.log(np.array(normwmean_grad_mod["layer 4"])),
            # mean_acts["last layer"]
            ])
        
        plotting(data=gradlistmean, name = "figures/grads_layer_seq_norm.png")
    
    def normwmean(self):
        
        assert self.grads["layer 0"].shape == (128,)
        epsilon = 1e-5
        gradlistmean = np.array([
            np.log(np.array(self.grads["layer 0"]) - np.mean(np.array(self.grads["layer 0"]), axis = 0) + epsilon),
            np.log(np.array(self.grads["layer 1"]) - np.mean(np.array(self.grads["layer 1"]), axis = 0) + epsilon),
            np.log(np.array(self.grads["layer 2"]) - np.mean(np.array(self.grads["layer 2"]), axis = 0) + epsilon),
            np.log(np.array(self.grads["layer 3"]) - np.mean(np.array(self.grads["layer 3"]), axis = 0) + epsilon),
            np.log(np.array(self.grads["layer 4"]) - np.mean(np.array(self.grads["layer 4"]), axis = 0) + epsilon),
        ])  
        
        plotting(data=gradlistmean, name = "figures/grad_layer_seq_normwmean.png")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default="8")
    
    args = parser.parse_args()
    
    
    # model = LanguageModel("EleutherAI/pythia-70m", device_map=t.device("cuda" if t.cuda.is_available() else "mps"))
    model = LanguageModel("EleutherAI/pythia-70m", device_map="cpu")
    train_data, val_data = inspect_data(data)
    print();print(model);print()
    
    try:
        with open(f"data/val_data_b{args.batch_size}.pkl", "rb") as f:
            val_dataloader = pickle.load(f)

    except:
        val_dataloader = process_data(model, train_data, val_data, args.batch_size)
        
        with open(f"data/val_data_b{args.batch_size}.pkl", "wb") as f:
            pickle.dump(val_dataloader, f)
    
    try:
        with open("data/activation_embeds_prenorm.pkl", "rb") as f:
            activation_embeds = pickle.load(f)
    except:
        activation_embeds = activation_embeds_fn(model, val_dataloader, args.batch_size)
        with open("data/activation_embeds_prenorm.pkl", "wb") as f:
            pickle.dump(activation_embeds, f)
    

    normed_class = normed(activation_embeds)
    normed_class.normwmean()
    normed_class.norm()
    
    normed_grad = gradients_norm(model, val_dataloader)
    normed_grad.norm()
    normed_grad.normwmean()
    