from imports import *

# Load the Pythia model and tokenizer
model_name = 'EleutherAI/pythia-70m'
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Prepare inputs
input_text = "The quick brown fox jumps over the lazy dog."
tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Set the model to training mode to ensure gradients are tracked
model.train()

# Define an input sequence
input_text = "Sample input text for gradient extraction."
inputs = tokenizer(input_text, return_tensors="pt").to(device)

class grad:
    
    def __init__(self,
                dataloader):
        self.dataloader = dataloader

    def forward(self):
        
        final_data_mlp = []
        final_data_attn = []
        data_attn = {
            "layer 0" : [],
            "layer 1" : [],
            "layer 2" : [],
            "layer 3" : [],
            "layer 4" : [],
            "layer 5" : [],
        }
        data_mlp = {
            "layer 0" : [],
            "layer 1" : [],
            "layer 2" : [],
            "layer 3" : [],
            "layer 4" : [],
            "layer 5" : [],
        }

        for batch in tqmd(self.dataloader):
            # Perform a forward pass to obtain the model output
            outputs = model(**batch)

            # Calculate the loss using any loss function, here we take sum of logits for simplicity
            loss = outputs.logits.sum()
            loss.backward()  # Backpropagate to calculate gradients

            # Extract the gradients for mlp.dense_4h_to_h and layers.0.attention.dense

            # Gradient for mlp.dense_4h_to_h
            mlp_grad_dict = {}
            attn_grad_dict = {}

            for layer in range(6):
                mlp_grad_dict[f"layer {layer}"] = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.grad

            for layer_ in range(6):
                attn_grad_dict[f"layer {layer_}"] = model.gpt_neox.layers[layer_].attention.dense.weight.grad


            for k, v in mlp_grad_dict.items():
                print(k,v)
                print()

            print()

            for key, value in attn_grad_dict.items():
                print(key,value)
                print()

            for layer in range(6):
                data_attn[f"layer {layer}"].append(t.mean(t.norm(attn_grad_dict[f"layer {layer}"], dim = -1), dim = 0))
                
            for layer in range(6):
                data_mlp[f"layer {layer}"].append(t.mean(t.norm(mlp_grad_dict[f"layer {layer}"], dim = -1), dim = 0))

        final_data_mlp = np.array([np.mean(np.array(data_mlp["layer 0"]), axis = 0),
                                   np.mean(np.array(data_mlp["layer 1"]), axis = 0),
                                   np.mean(np.array(data_mlp["layer 2"]), axis = 0),
                                   np.mean(np.array(data_mlp["layer 3"]), axis = 0),
                                   np.mean(np.array(data_mlp["layer 4"]), axis = 0),
        ])
        final_data_attn = np.array([np.mean(np.array(data_attn["layer 0"]), axis = 0),
                                   np.mean(np.array(data_attn["layer 1"]), axis = 0),
                                   np.mean(np.array(data_attn["layer 2"]), axis = 0),
                                   np.mean(np.array(data_attn["layer 3"]), axis = 0),
                                   np.mean(np.array(data_attn["layer 4"]), axis = 0),
        ])
        
        with open("mdata/pythia_mlp_gradient.pkl", "wb") as f:
            pickle.dump(final_data_mlp, f)
        
        with open("mdata/pythia_attn_gradient.pkl", "wb") as p:
            pickle.dump(final_data_attn,p)
        
        self.plotting(final_data_attn, "attention", "Attention")
        self.plotting(final_data_mlp, "mlp", "MLP")

    def plotting(self, data, name, title):
        fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
        cax = ax.imshow(data, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

        # Add color bar to indicate the scale
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Gradient Magnitude')

        # Set labels
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Layers')

        # Set x-ticks and y-ticks
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels([f'Token {i}' for i in range(data.shape[1])])
        ax.set_yticklabels(["layer 0", "layer 1", "layer 2", "layer 3", "layer 4", "layer 5"])

        # Optionally, you can add titles
        plt.title(f"[Pythia]: Gradient of Pythia {title} Output")

        # Save the plot
        plt.savefig(f"mfigures/Pythia_grad_{name}.png")
        plt.close()

if __name__ == "__main__":
    
    with open(f"data/pythia_val_data_b16.pkl", "rb") as f:
        val_dataloader = pickle.load(f)
    
    gradient_plot = grad(val_dataloader)
    gradient_plot.forward()