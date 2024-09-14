from imports import *

class pythia_grad:
    
    def __init__(self, model, tokenizer, dataloader, outputs):
        self.model = model
        self.tokenizer = tokenizer
        self.dataloader = dataloader
        self.outputs = outputs
        self.mlp_gradients = None
        self.attn_gradients = None
        self.layer_gradients = {}


    def capture_gradients(self, layer_name):
        def hook(module, grad_input, grad_output):
            self.layer_gradients[layer_name] = grad_output[0]
        return hook
    
    
    def forward(self):
        for batch in self.dataloader:
            inputs = self.tokenizer(batch, return_tensors="pt")
            outputs = self.model(**inputs)

            for i, layer in enumerate(self.model.h):
                layer.mlp.register_backward_hook(self.capture_gradients(f"h[{i}].mlp"))
                layer.attn.register_backward_hook(self.capture_gradients(f"h[{i}].attn"))
            
            outputs = self.model(**inputs)
            last_hidden_states = outputs.last_hidden_state

            # Compute a loss based on the last hidden states
            loss = last_hidden_states.sum()

            # Backward pass to compute the gradients
            loss.backward()

            # Print captured gradients
            for layer_name, grad in self.layer_gradients.items():
                print(f"Gradient for {layer_name}:", grad)

    
    def plot_grad():
        pass