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

# Perform a forward pass to obtain the model output
outputs = model(**inputs)

# Calculate the loss using any loss function, here we take sum of logits for simplicity
loss = outputs.logits.sum()
loss.backward()  # Backpropagate to calculate gradients

# Extract the gradients for mlp.dense_4h_to_h and layers.0.attention.dense

# Gradient for mlp.dense_4h_to_h
mlp_grad_dict = {}
attn_grad_dict = {}

for layer in range(6):
    mlp_grad_dict[f" layer {layer}"] = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight.grad

for layer_ in range(6):
    attn_grad_dict[f"layer {layer}"] = model.gpt_neox.layers[layer].attention.dense.weight.grad


for k, v in mlp_grad_dict.items():
    print(k,v)
    print()

print()

for key, value in attn_grad_dict.items():
    print(key,value)
    print()

