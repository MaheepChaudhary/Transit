from imports import *


class Gradient_MLP:
    
    def __init__(self, data, device, dataset_name, model_name):
        
        self.model_name = model_name
        
        if self.model_name == "Pythia14m":
            self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-14m')
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-14m')
            
        elif model_name == "Pythia70m":
            self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m')
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
            
        elif model_name == "Pythia160m":
            self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
        
        elif model_name == "Pythia410m":
            self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-410m')
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-410m')
        
        elif model_name == "Pythia1b":
            self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b')
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')
        
        elif model_name == "Pythia1.4b":
            self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1.4b')
            self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1.4b')
        
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.data = data
        self.device = device
        self.dataset_name = dataset_name
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
        self.model.to(device)
        self.model.gpt_neox.embed_in.requires_grad = False
        for layer in self.model.gpt_neox.layers:
            for param in layer.parameters():
                param.requires_grad = False
            for param in layer.mlp.dense_4h_to_h.parameters():
                param.requires_grad = True
            for param in layer.attention.dense.parameters():
                param.requires_grad = True

        
    def forward(self):
        
        final_data = []
        attn_final_data = []
        
        for sample in tqdm(self.data):
    
            token_gradients = []
            attn_token_gradients = []
            
            inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)

            # Get the outputs and compute loss
            outputs = self.model(**inputs)
            logits = outputs.logits
            loss = logits.sum()  # Example loss
            
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
                attn_gradients = []
                for i in range(6):
                    layer = self.model.gpt_neox.layers[i].mlp.dense_4h_to_h  # Access the specific layer/parameter (adapted to Pythia)
                    attn_layer = self.model.gpt_neox.layers[i].attention.dense  # Access the specific layer/parameter (adapted to Pythia)
                    attn_param_grad = attn_layer.weight.grad.clone().view(-1)  # Clone and reshape the gradient
                    param_grad = layer.weight.grad.clone().view(-1)  # Clone and reshape the gradient
                    gradients.append(param_grad.unsqueeze(0))  # Append the gradient for this token
                    attn_gradients.append(attn_param_grad.unsqueeze(0))  # Append the gradient for this token

                # Convert gradients to a tensor and add to the list
                token_gradients.append(torch.cat(gradients, dim=0))  # Shape: (layer_count, output_dim)
                attn_token_gradients.append(torch.cat(attn_gradients, dim=0))  # Shape: (layer_count, output_dim)

            # Convert token gradients to a tensor for visualization
            token_gradients_tensor = torch.stack(token_gradients)  # Shape: (seq_len, layer_count, output_dim)
            attn_token_gradients_tensor = torch.stack(attn_token_gradients)  # Shape: (seq_len, layer_count, output_dim)
            
            # Compute the average gradient norm across layers for visualization
            average_gradients_tensor = torch.log(token_gradients_tensor.norm(dim=2))  # Shape: (seq_len, layer_count)
            average_attn_gradients_tensor = torch.log(attn_token_gradients_tensor.norm(dim=2))  # Shape: (seq_len, layer_count)

        final_data.append(average_gradients_tensor)
        attn_final_data.append(average_attn_gradients_tensor)
            
        try:
            os.mkdir(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        
        with open(f"data/{self.dataset_name}/{self.model_name}/gradient_mlp.pkl", "wb") as f:
            pickle.dump(final_data, f)
            
        with open(f"data/{self.dataset_name}/{self.model_name}/gradient_attention.pkl", "wb") as f:
            pickle.dump(attn_final_data, f)

        self.visualise(final_data, average_gradients_tensor, name = "MLP", title = "mlp.dense_4h_to_h")
        self.visualise(attn_final_data, average_attn_gradients_tensor, name = "Attention", title = "attention.dense")


    def visualise(self, final_data, average_gradients_tensor, name, title):

        # Visualize the gradients using a heatmap
        plt.figure(figsize=(15, 6))
        sns.heatmap(np.mean(np.array(final_data), axis = 0).T, cmap='viridis', cbar=True, yticklabels=range(6), xticklabels=range(average_gradients_tensor.size(0)))
        plt.xlabel('Token Index')
        plt.ylabel('Layer Index')
        plt.title(f'[{self.model_name}-{self.dataset_name}]Token-wise Gradient for {title} on log scale')
        plt.savefig(f"figures/{self.dataset_name}/{self.model_name}/gradient_{name}.png")


# class Gradient_attn:
    
#     def __init__(self, data, device, dataset_name, model_name):
        
#         self.model_name = model_name
        
#         if self.model_name == "Pythia14m":
#             self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-14m')
#             self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-14m')
            
#         elif model_name == "Pythia70m":
#             self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-70m')
#             self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')
            
#         elif model_name == "Pythia160m":
#             self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-160m')
#             self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-160m')
        
#         elif model_name == "Pythia410m":
#             self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-410m')
#             self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-410m')
        
#         elif model_name == "Pythia1b":
#             self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1b')
#             self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1b')
        
#         elif model_name == "Pythia1.4b":
#             self.model = AutoModelForCausalLM.from_pretrained('EleutherAI/pythia-1.4b')
#             self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-1.4b')
            
#         self.tokenizer.pad_token = self.tokenizer.eos_token
        
#         self.data = data
#         self.device = device
#         self.dataset_name = dataset_name
        
#         if self.dataset_name == "tinystories":
#             self.max_length = 145
#         elif self.dataset_name == "summarisation":
#             self.max_length = 340
#         elif self.dataset_name == "alpaca":
#             self.max_length = 10
        
#     def forward(self):
        
#         final_data = []
        
#         for sample in tqdm(self.data):
    
#             token_gradients = []
            
#             inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)

#             # Get the outputs and compute loss
#             outputs = self.model(**inputs)
#             logits = outputs.logits
#             loss = logits.sum()  # Example loss

#             # Iterate over each token
#             for token_idx in range(inputs["input_ids"].shape[1]):  # Loop over the sequence length (tokens)
#                 self.model.zero_grad()  # Clear any previous gradients
                
#                 # Compute loss only for this specific token's contribution
#                 # Modify this if needed to focus on the exact component of the loss related to the token
#                 token_loss = logits[0, token_idx, :].sum()
                
#                 # Perform backward pass
#                 token_loss.backward(retain_graph=True)  # retain_graph=True allows subsequent backward passes
                
#                 # Collect the gradient of the specific parameter for this token
#                 gradients = []
#                 for i in range(6):
#                     layer = self.model.gpt_neox.layers[i].attention.dense  # Access the specific layer/parameter (adapted to Pythia)
#                     param_grad = layer.weight.grad.clone().view(-1)  # Clone and reshape the gradient
#                     gradients.append(param_grad.unsqueeze(0))  # Append the gradient for this token

#                 # Convert gradients to a tensor and add to the list
#                 token_gradients.append(torch.cat(gradients, dim=0))  # Shape: (layer_count, output_dim)

#             # Convert token gradients to a tensor for visualization
#             token_gradients_tensor = torch.stack(token_gradients)  # Shape: (seq_len, layer_count, output_dim)

#             # Compute the average gradient norm across layers for visualization
#             average_gradients_tensor = torch.log(token_gradients_tensor.norm(dim=2))  # Shape: (seq_len, layer_count)

#             final_data.append(average_gradients_tensor)
            
#         try:
#             os.mkdir(f"data/{self.dataset_name}/{self.model_name}")
#         except:
#             pass
        
#         with open(f"data/{self.dataset_name}/{self.model_name}/gradient_attention.pkl", "wb") as f:
#             pickle.dump(final_data, f)

#         self.visualise(final_data, average_gradients_tensor)


#     def visualise(self, final_data, average_gradients_tensor):

#         # Visualize the gradients using a heatmap
#         plt.figure(figsize=(15, 6))
#         sns.heatmap(np.mean(np.array(final_data), axis = 0).T, cmap='viridis', cbar=True, yticklabels=range(6), xticklabels=range(average_gradients_tensor.size(0)))
#         plt.xlabel('Token Index')
#         plt.ylabel('Layer Index')
#         plt.title(f'[{self.model_name}-{self.dataset_name}]Token-wise Gradient for attention.dense on log scale')
#         plt.savefig(f"figures/{self.dataset_name}/{self.model_name}/gradient_attention.png")