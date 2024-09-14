from imports import *

class pythia_grad:
    
    def __init__(self, model, tokenizer, dataloader):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.mlp_gradients = None
        self.attn_gradients = None
        self.layer_gradients = {}
        self.mlp_dict = {
            "layers[5].mlp": [],
            "layers[4].mlp": [],
            "layers[3].mlp": [],
            "layers[2].mlp": [],
            "layers[1].mlp": [],
            "layers[0].mlp": [],
        }
        self.attn_dict = {
            "h[5].attention": [],
            "h[4].attention": [],
            "h[3].attention": [],
            "h[2].attention": [],
            "h[1].attention": [],
            "h[0].attention": []
        }
        self.resid_dict = {
            "resid[5]": [],
            "resid[4]": [],
            "resid[3]": [],
            "resid[2]": [],
            "resid[1]": [],
            "resid[0]": []
        }


    def capture_gradients(self, layer_name):
        def hook(module, grad_input, grad_output):
            self.layer_gradients[layer_name] = grad_output[0]
        return hook
    
    
    def forward(self):
        for j, batch in enumerate(tqdm(self.dataloader)):
            # Assume batch is already tokenized; adjust as necessary
            inputs = {key: t.tensor(val) for key, val in batch.items()}

            for i, layer in enumerate(self.model.gpt_neox.layers):
                # Register backward hooks for MLP and attention gradients
                layer.mlp.register_backward_hook(self.capture_gradients(f"layers[{i}].mlp", self.mlp_dict))
                layer.attention.register_backward_hook(self.capture_gradients(f"h[{i}].attention", self.attn_dict))
            
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Compute a loss based on the logits
            loss = logits.sum()

            # Backward pass to compute the gradients
            loss.backward()

            # Initialize storage for current batch gradients
            current_mlp_grad = {key: [] for key in self.mlp_dict.keys()}
            current_attn_grad = {key: [] for key in self.attn_dict.keys()}
            current_resid_grad = {key: [] for key in self.resid_dict.keys()}

            # Compute and store the gradients for the current batch
            for i in range(len(self.model.gpt_neox.layers)):
                mlp_grad = self.mlp_dict[f"layers[{i}].mlp"]
                attn_grad = self.attn_dict[f"h[{i}].attention"]

                # Compute norms and means separately
                mlp_grad_mean = t.mean(t.norm(mlp_grad, dim=-1), dim=0)
                attn_grad_mean = t.mean(t.norm(attn_grad, dim=-1), dim=0)
                
                # Compute residual gradients and their mean
                resid_grad = mlp_grad + attn_grad
                resid_grad_mean = t.mean(t.norm(resid_grad, dim=-1), dim=0)

                # Store gradients in the current batch storage
                current_mlp_grad[f"layers[{i}].mlp"].append(mlp_grad_mean)
                current_attn_grad[f"h[{i}].attention"].append(attn_grad_mean)
                current_resid_grad[f"resid[{i}]"].append(resid_grad_mean)
            
            # Update dictionaries with the mean of gradients over batches
            for layer_name in self.mlp_dict.keys():
                self.mlp_dict[layer_name] = np.mean(np.array(current_mlp_grad[layer_name]), axis=0)

            for layer_name in self.attn_dict.keys():
                self.attn_dict[layer_name] = np.mean(np.array(current_attn_grad[layer_name]), axis=0)

            for layer_name in self.resid_dict.keys():
                self.resid_dict[layer_name] = np.mean(np.array(current_resid_grad[layer_name]), axis=0)

            if j == 100:
                break

        # Plot the results
        self.plotting_grad(self.mlp_dict, "MLP")
        self.plotting_grad(self.attn_dict, "Attention")
        self.plotting_grad(self.resid_dict, "Residual")

    def plotting_grad(self, data_dict, name):
        # Convert dictionary values to a list and sort them by key
        data = np.array([np.mean(v, axis=0) for k, v in sorted(data_dict.items())])

        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
        cax = ax.imshow(data, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

        # Add color bar to indicate the scale
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Gradient Magnitude')

        # Set labels
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Layers')

        # Set x-ticks and y-ticks
        num_layers = len(data_dict)
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(num_layers))
        ax.set_xticklabels([f'Token {i}' for i in range(data.shape[1])])
        ax.set_yticklabels(list(sorted(data_dict.keys())))

        # Optionally, you can add titles
        plt.title(f"[Pythia]: Gradient of {name}")

        # Ensure the 'mfigures' directory exists
        plt.savefig(f"mfigures/{name}.png")
        plt.close()
