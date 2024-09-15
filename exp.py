from imports import *

# Load the GPT-2 model and tokenizer
model_name = 'openai-community/gpt2'  # Change this to the specific GPT-2 model you want to use
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Prepare inputs
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Set the model to training mode
model.train()

# Initialize variables to store gradients
mlp_gradients = None
attn_gradients = None

# Hook functions to capture gradients
def capture_mlp_gradients(module, grad_in, grad_out):
    global mlp_gradients
    mlp_gradients = grad_out[0]  # Store the gradients of MLP output

def capture_attn_gradients(module, grad_in, grad_out):
    global attn_gradients
    attn_gradients = grad_out[0]  # Store the gradients of attention output

# Register hooks for the first MLP and attention layers in GPT-2
model.transformer.h[0].mlp.c_fc.register_full_backward_hook(capture_mlp_gradients)
model.transformer.h[0].attn.c_proj.register_full_backward_hook(capture_attn_gradients)

# Forward pass through the model
outputs = model(**inputs)
logits = outputs.logits

# Calculate a dummy loss and perform backpropagation
loss = logits.sum()  # Example loss function (sum of logits)
loss.backward()

# Print the captured gradients
print("MLP Gradients:", mlp_gradients)
print("Attention Gradients:", attn_gradients)
