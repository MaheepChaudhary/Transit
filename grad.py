from imports import *


class Gradient_MLP:
    
    def __init__(self, model, data, device, tokenizer, dataset_name, model_name):
        self.model = model
        self.data = data
        self.device = device
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.model_name = model_name
        
    def forward(self):
        
        final_data = []
        
        for sample in tqdm(self.data):
    
            token_gradients = []
            
            inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=128, truncation=True)

            # Get the outputs and compute loss
            outputs = self.model(**inputs)
            logits = outputs.logits
            loss = logits.sum()  # Example loss

            print(logits.shape)

            # Iterate over each token
            for token_idx in range(inputs["input_ids"].shape[1]):  # Loop over the sequence length (tokens)
                self.model.zero_grad()  # Clear any previous gradients
                
                # Compute loss only for this specific token's contribution
                # Modify this if needed to focus on the exact component of the loss related to the token
                token_loss = logits[0, token_idx, :].sum()
                
                # Perform backward pass
                token_loss.backward(retain_graph=True)  # retain_graph=True allows subsequent backward passes
                
                # Collect the gradient of the specific parameter for this token
                gradients = []
                for i in range(6):
                    layer = self.model.gpt_neox.layers[i].mlp.dense_4h_to_h  # Access the specific layer/parameter (adapted to Pythia)
                    param_grad = layer.weight.grad.clone().view(-1)  # Clone and reshape the gradient
                    gradients.append(param_grad.unsqueeze(0))  # Append the gradient for this token

                # Convert gradients to a tensor and add to the list
                token_gradients.append(torch.cat(gradients, dim=0))  # Shape: (layer_count, output_dim)

            # Convert token gradients to a tensor for visualization
            token_gradients_tensor = torch.stack(token_gradients)  # Shape: (seq_len, layer_count, output_dim)

            # Compute the average gradient norm across layers for visualization
            average_gradients_tensor = torch.log(token_gradients_tensor.norm(dim=2))  # Shape: (seq_len, layer_count)

            final_data.append(average_gradients_tensor)

        self.visualise(final_data, average_gradients_tensor)


    def visualise(self, final_data, average_gradients_tensor):

        # Visualize the gradients using a heatmap
        plt.figure(figsize=(15, 6))
        sns.heatmap(np.mean(np.array(final_data), axis = 0).T, cmap='viridis', cbar=True, yticklabels=range(6), xticklabels=range(average_gradients_tensor.size(0)))
        plt.xlabel('Token Index')
        plt.ylabel('Layer Index')
        plt.title(f'[{self.model_name}-{self.dataset_name}]Token-wise Gradient for mlp.dense_4h_to_h on log scale')
        plt.savefig(f"figures/{self.dataset_name}/{self.model_name}/activation_mlp.png")


class Gradient_attn:
    
    def __init__(self, model, data, device, tokenizer, dataset_name, model_name):
        self.model = model
        self.data = data
        self.device = device
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.model_name = model_name
        
    def forward(self):
        
        final_data = []
        
        for sample in tqdm(self.data):
    
            token_gradients = []
            
            inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=128, truncation=True)

            # Get the outputs and compute loss
            outputs = self.model(**inputs)
            logits = outputs.logits
            loss = logits.sum()  # Example loss

            print(logits.shape)

            # Iterate over each token
            for token_idx in range(inputs["input_ids"].shape[1]):  # Loop over the sequence length (tokens)
                self.model.zero_grad()  # Clear any previous gradients
                
                # Compute loss only for this specific token's contribution
                # Modify this if needed to focus on the exact component of the loss related to the token
                token_loss = logits[0, token_idx, :].sum()
                
                # Perform backward pass
                token_loss.backward(retain_graph=True)  # retain_graph=True allows subsequent backward passes
                
                # Collect the gradient of the specific parameter for this token
                gradients = []
                for i in range(6):
                    layer = self.model.gpt_neox.layers[i].attention.dense  # Access the specific layer/parameter (adapted to Pythia)
                    param_grad = layer.weight.grad.clone().view(-1)  # Clone and reshape the gradient
                    gradients.append(param_grad.unsqueeze(0))  # Append the gradient for this token

                # Convert gradients to a tensor and add to the list
                token_gradients.append(torch.cat(gradients, dim=0))  # Shape: (layer_count, output_dim)

            # Convert token gradients to a tensor for visualization
            token_gradients_tensor = torch.stack(token_gradients)  # Shape: (seq_len, layer_count, output_dim)

            # Compute the average gradient norm across layers for visualization
            average_gradients_tensor = torch.log(token_gradients_tensor.norm(dim=2))  # Shape: (seq_len, layer_count)

            final_data.append(average_gradients_tensor)

        self.visualise(final_data, average_gradients_tensor)


    def visualise(self, final_data, average_gradients_tensor):

        # Visualize the gradients using a heatmap
        plt.figure(figsize=(15, 6))
        sns.heatmap(np.mean(np.array(final_data), axis = 0).T, cmap='viridis', cbar=True, yticklabels=range(6), xticklabels=range(average_gradients_tensor.size(0)))
        plt.xlabel('Token Index')
        plt.ylabel('Layer Index')
        plt.title(f'[{self.model_name}-{self.dataset_name}]Token-wise Gradient for mlp.dense_4h_to_h on log scale')
        plt.savefig(f"figures/{self.dataset_name}/{self.model_name}/activation_mlp.png")