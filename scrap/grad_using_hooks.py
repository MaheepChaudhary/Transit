from scrap.imports import *

class pythia_grad:
    
    def __init__(self, model, tokenizer, dataloader):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.final_mlp_dict = {
            "layers[5].mlp": [],
            "layers[4].mlp": [],
            "layers[3].mlp": [],
            "layers[2].mlp": [],
            "layers[1].mlp": [],
            "layers[0].mlp": [],
        }
        self.final_attn_dict = {
            "layers[5].attention": [],
            "layers[4].attention": [],
            "layers[3].attention": [],
            "layers[2].attention": [],
            "layers[1].attention": [],
            "layers[0].attention": []
        }
        self.final_resid_dict = {
            "resid[5]": [],
            "resid[4]": [],
            "resid[3]": [],
            "resid[2]": [],
            "resid[1]": [],
            "resid[0]": []
        }
    
    def capture_gradients(self, layer_name, grad_dict):
        def hook(module, grad_input, grad_output):
            # Initialize list if not present
            if layer_name not in grad_dict:
                grad_dict[layer_name] = []
            grad_dict[layer_name].append(grad_output[0].clone().detach())
        return hook
    
    def forward(self):
        
        current_mlp_grad = {key: [] for key in self.final_mlp_dict.keys()}
        current_attn_grad = {key: [] for key in self.final_attn_dict.keys()}
        current_resid_grad = {key: [] for key in self.final_resid_dict.keys()}
        
        for j, batch in enumerate(tqdm(self.dataloader)):
            
            mlp_dict = {
            "layers[5].mlp": [],
            "layers[4].mlp": [],
            "layers[3].mlp": [],
            "layers[2].mlp": [],
            "layers[1].mlp": [],
            "layers[0].mlp": [],
            }
            attn_dict = {
                "layers[5].attention": [],
                "layers[4].attention": [],
                "layers[3].attention": [],
                "layers[2].attention": [],
                "layers[1].attention": [],
                "layers[0].attention": []
            }
            resid_dict = {
                "resid[5]": [],
                "resid[4]": [],
                "resid[3]": [],
                "resid[2]": [],
                "resid[1]": [],
                "resid[0]": []
            }
            # Convert batch to tensor if not already; adjust as necessary
            inputs = {key: t.tensor(val) for key, val in batch.items()}

            # Register hooks
            for i, layer in enumerate(self.model.gpt_neox.layers):
                layer.mlp.register_backward_hook(self.capture_gradients(f"layers[{i}].mlp", mlp_dict))
                layer.attention.register_backward_hook(self.capture_gradients(f"layers[{i}].attention", attn_dict))

            outputs = self.model(**inputs)
            logits = outputs.logits

            # Compute a loss based on the logits
            loss = logits.sum()

            # Backward pass to compute the gradients
            loss.backward()

            # Compute and store gradients


            for i in range(len(self.model.gpt_neox.layers)):
                # Stack and convert lists of gradients to tensors
                mlp_grad = t.stack(mlp_dict[f"layers[{i}].mlp"]).squeeze(0)
                attn_grad = t.stack(attn_dict[f"layers[{i}].attention"]).squeeze(0)

                # print(f"MLP grad: {np.array(mlp_grad).shape}")
                # Compute norms and means separately
                mlp_grad_mean = t.mean(t.norm(mlp_grad, dim=-1), dim=0)
                attn_grad_mean = t.mean(t.norm(attn_grad, dim=-1), dim=0)
                print(mlp_grad_mean)
                # print(f"MLP Grad mean: {np.array(mlp_grad_mean).shape}")
                # Compute residual gradients and their mean
                resid_grad = mlp_grad + attn_grad
                resid_grad_mean = t.mean(t.norm(resid_grad, dim=-1), dim=0)

                # print(f"Resid Grad: {np.array(resid_grad_mean).shape}")
                print()
                # Store gradients in the current batch storage
                current_mlp_grad[f"layers[{i}].mlp"].append(mlp_grad_mean)
                current_attn_grad[f"layers[{i}].attention"].append(attn_grad_mean)
                current_resid_grad[f"resid[{i}]"].append(resid_grad_mean)
                
            if j == 5:
                break
            
        # Update dictionaries with the mean of gradients over batches
        for layer_name in self.final_mlp_dict.keys():
            if current_mlp_grad[layer_name]:
                self.final_mlp_dict[layer_name] = np.mean(np.array(current_mlp_grad[layer_name]), axis=0)

        for layer_name in self.final_attn_dict.keys():
            if current_attn_grad[layer_name]:
                self.final_attn_dict[layer_name] = np.mean(np.array(current_attn_grad[layer_name]), axis=0)

        for layer_name in self.final_resid_dict.keys():
            if current_resid_grad[layer_name]:
                self.final_resid_dict[layer_name] = np.mean(np.array(current_resid_grad[layer_name]), axis=0)
        

            
        # Plot the results
        self.plotting_grad(self.final_mlp_dict, "MLP")
        self.plotting_grad(self.final_attn_dict, "Attention")
        self.plotting_grad(self.final_resid_dict, "Residual")

    def plotting_grad(self, data_dict, name):
        data = np.array([v for k, v in sorted(data_dict.items())])

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
        ax.set_xticks(np.arange(data.shape[1]))
        ax.set_yticks(np.arange(data.shape[0]))
        ax.set_xticklabels([f'Token {i}' for i in range(data.shape[1])])
        ax.set_yticklabels(list(sorted(data_dict.keys())))

        # Optionally, you can add titles
        plt.title(f"[Pythia]: Gradient of Pythia {name} Output")

        # Save the plot
        plt.savefig(f"mfigures/{name}.png")
        plt.close()
