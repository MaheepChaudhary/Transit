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

class single_sample_act_norm:
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def activation(self):
        
        act_dict = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        random_samples = random.sample(self.data["text"], 20)
        for index, sample in enumerate(random_samples):
            with self.model.trace(sample) as tracer:
                output0 = self.model.gpt_neox.layers[0].mlp.act.output.save()
                output1 = self.model.gpt_neox.layers[1].mlp.act.output.save()
                output2 = self.model.gpt_neox.layers[2].mlp.act.output.save()
                output3 = self.model.gpt_neox.layers[3].mlp.act.output.save()
                output4 = self.model.gpt_neox.layers[4].mlp.act.output.save()
                output = self.model.embed_out.output.save()
            
            for i,j in enumerate([output0, output1, output2, output3, output4]):    
                act_dict[f"layer {i}"].append(np.array([t.norm(j, dim = -1).detach()]))
            
        return act_dict
    
    def norm(self):
        
        activations = self.activation()
        for index in range(len(activations["layer 0"])):
            data = np.array(
                [activations["layer 0"][index].squeeze(0).squeeze(0),
                activations["layer 1"][index].squeeze(0).squeeze(0),
                activations["layer 2"][index].squeeze(0).squeeze(0),
                activations["layer 3"][index].squeeze(0).squeeze(0),
                activations["layer 4"][index].squeeze(0).squeeze(0)]
            )
                
            self.plot(data=data, name=f"mfigures_norm/single_sample_{index}.png")
    
    def plot(self, data, name):
        plt.figure(figsize=(10, 5))
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        plt.title('Single Sample Token activations in different layers')
        plt.savefig(name)
        plt.close()
            

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
    if name == "figures/layer_seq_norm.png":
        plt.title('Token activations in different layers')
    elif name == "figures/layer_seq_normwmean.png":
        plt.title('Diff Mean Token activations in different layers')
    elif name == "figures/grads_layer_seq_norm.png":
        plt.title('Token gradients in different layers')
    elif name == "figures/grad_layer_seq_normwmean.png":
        plt.title('Diff Mean Token gradients in different layers')

    # Show the heatmap
    plt.savefig(name)
    plt.close()
    


class single_sample_grad_norm:
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def gradients(self):
        
        act_dict = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        random_samples = random.sample(self.data["text"], 20)
        for index, sample in enumerate(random_samples):
            with self.model.trace(sample) as tracer:
                output0 = self.model.gpt_neox.layers[0].output.mlp.act.grad.save()
                output1 = self.model.gpt_neox.layers[1].output.mlp.act.grad.save()
                output2 = self.model.gpt_neox.layers[2].output.mlp.act.grad.save()
                output3 = self.model.gpt_neox.layers[3].output.mlp.act.grad.save()
                output4 = self.model.gpt_neox.layers[4].output.mlp.act.grad.save()
            
            for i,j in enumerate([output0, output1, output2, output3, output4]):
                print(j)    
                act_dict[f"layer {i}"].append(np.array([t.norm(j, dim = -1).detach()]))

        return act_dict
    
    def grad_norm(self):
        
        gradients = self.gradients()
        for index in range(len(gradients["layer 0"])):
            grad_data = np.array(
                [gradients["layer 0"][index].squeeze(0).squeeze(0),
                gradients["layer 1"][index].squeeze(0).squeeze(0),
                gradients["layer 2"][index].squeeze(0).squeeze(0),
                gradients["layer 3"][index].squeeze(0).squeeze(0),
                gradients["layer 4"][index].squeeze(0).squeeze(0)]
            )
                
            self.plot(data=grad_data, name=f"mfigures_gradnorm/single_sample_{index}.png")
    
    def plot(self, data, name):
        plt.figure(figsize=(10, 5))
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        plt.title('Single Sample Token activations in different layers')
        plt.savefig(name)
        plt.close()
            

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
    if name == "figures/layer_seq_norm.png":
        plt.title('Token activations in different layers')
    elif name == "figures/layer_seq_normwmean.png":
        plt.title('Diff Mean Token activations in different layers')
    elif name == "figures/grads_layer_seq_norm.png":
        plt.title('Token gradients in different layers')
    elif name == "figures/grad_layer_seq_normwmean.png":
        plt.title('Diff Mean Token gradients in different layers')

    # Show the heatmap
    plt.savefig(name)
    plt.close()


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
            with open("data/grads_resid.pkl", "rb") as f:
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
            
                output0 = self.model.gpt_neox.layers[0].output[0].grad.save()
                output1 = self.model.gpt_neox.layers[1].output[0].grad.save()
                output2 = self.model.gpt_neox.layers[2].output[0].grad.save()
                output3 = self.model.gpt_neox.layers[3].output[0].grad.save()
                output4 = self.model.gpt_neox.layers[4].output[0].grad.save()
                
                self.model.output.logits.sum().backward()
            
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            grad_embeds["layer 0"].append(t.norm(t.norm(output0, dim = 0), dim = -1))
            grad_embeds["layer 1"].append(t.norm(t.norm(output1, dim = 0), dim = -1))
            grad_embeds["layer 2"].append(t.norm(t.norm(output2, dim = 0), dim = -1))
            grad_embeds["layer 3"].append(t.norm(t.norm(output3, dim = 0), dim = -1))
            grad_embeds["layer 4"].append(t.norm(t.norm(output4, dim = 0), dim = -1))
            
        with open("data/grads_resid.pkl", "wb") as f:
            pickle.dump(grad_embeds, f)
            
        return grad_embeds

    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        grad_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        
        grad_actemb["layer 0"] = np.linalg.norm(self.grads["layer 0"], axis=0)
        grad_actemb["layer 1"] = np.linalg.norm(self.grads["layer 1"], axis=0)
        grad_actemb["layer 2"] = np.linalg.norm(self.grads["layer 2"], axis=0)
        grad_actemb["layer 3"] = np.linalg.norm(self.grads["layer 3"], axis=0)
        grad_actemb["layer 4"] = np.linalg.norm(self.grads["layer 4"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        gradlist = np.array([
            np.log(np.array(grad_actemb["layer 0"])),
            np.log(np.array(grad_actemb["layer 1"])),
            np.log(np.array(grad_actemb["layer 2"])),
            np.log(np.array(grad_actemb["layer 3"])),
            np.log(np.array(grad_actemb["layer 4"])),
            # mean_acts["last layer"]
            ])
        
        plotting(data=gradlist, name = "figures/grads_layer_seq_norm_grad_resid.png")
    
    def normwmean(self):
        
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
        
        normwmean_grad["layer 0"] = np.array(self.grads["layer 0"]) - np.mean(np.array(self.grads["layer 0"]), axis = 0)
        normwmean_grad["layer 1"] = np.array(self.grads["layer 1"]) - np.mean(np.array(self.grads["layer 1"]), axis = 0)
        normwmean_grad["layer 2"] = np.array(self.grads["layer 2"]) - np.mean(np.array(self.grads["layer 2"]), axis = 0)
        normwmean_grad["layer 3"] = np.array(self.grads["layer 3"]) - np.mean(np.array(self.grads["layer 3"]), axis = 0)
        normwmean_grad["layer 4"] = np.array(self.grads["layer 4"]) - np.mean(np.array(self.grads["layer 4"]), axis = 0)
        
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
        
        plotting(data=gradlistmean, name = "figures/grad_layer_seq_normwmean_grad_resid.png")


def img_concat():

    # List of image file paths (assuming you have 22 image paths)
    image_paths = ["mfigures_gradnorm/" + img for img in os.listdir("mfigures_gradnorm")]

    # Determine the grid size, for example, 5 rows and 5 columns
    rows = 5
    cols = 4

    # Create a figure to hold the grid
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop over images and axes and plot each image
    for i, (img_path, ax) in enumerate(zip(image_paths, axes)):
        if i < len(image_paths):
            img = mpimg.imread(img_path)
            ax.imshow(img)
        ax.axis('off')  # Hide the axes

    # Save the combined image or display it
    plt.tight_layout()
    plt.savefig("mfigures_gradnorm/combined_image.png")


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
    

    # normed_class = normed(activation_embeds)
    # normed_class.normwmean()
    # normed_class.norm()
    
    # normed_grad = gradients_norm(model, val_dataloader)
    # normed_grad.norm()
    # normed_grad.normwmean()
    
    # normed_single = single_sample_act_norm(model, val_data)
    # normed_single.norm()
    
    # img_concat()
    
    grad_normed_single = single_sample_grad_norm(model, val_data)
    grad_normed_single.grad_norm()
    
    img_concat()