from imports import *
from dataset import inspect_data

from transformers import GPT2Tokenizer, GPT2Model

# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')
print(model)

# Input text
input_text = "Hello, how are you?"
inputs = tokenizer(input_text, return_tensors="pt")

# Forward pass
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state  # This is the output of the GPT-2 model

# Suppose we want the gradient of the sum of the last hidden states (or any other component)
loss = last_hidden_states.sum()

# Backward pass to compute the gradients
loss.backward()

# Now, we can access the gradients
grad_input_ids = inputs['input_ids'].grad  # Gradient wrt input tokens

# So i need the gradient of the following parameters of the model:
'''
* model.h.0.mlp.c_proj.weight
* model.h.0.attn.c_proj.weight
* But there is no variable that can denote the additiona of the mlp + resid to get their gradient. 
'''

first_layer_gradients = model.h[0].mlp.c_fc.weight.grad
print(first_layer_gradients)
    # print(param)
# print(grad_input_ids)
# If you want gradients of specific layers or model parameters, you can access them:
# for name, param in model.named_parameters():
#     if param.requires_grad and param.grad is not None:
#         print(f"Gradient for {name}: {param.grad}")



'''
from nnsight import LanguageModel

model = LanguageModel("EleutherAI/pythia-410m", device_map="cpu")

a = model.tokenizer("I will be out for a ")

print(np.array(a["input_ids"]).shape)

# train_data, val_data = inspect_data(data)

# random_samples = ["I will be out for a" for i in range(20)]

# for index, sample in enumerate(random_samples):

tensor = t.rand(50304)
tensor = tensor.unsqueeze(0).unsqueeze(0)
tensor = tensor.expand(1, 6, 50304)

with model.trace("I will be out for a") as tracer:
    output0 = model.gpt_neox.layers[0].output[0].grad.save()
    output0_mlp = model.gpt_neox.layers[0].mlp.output.grad.save()
    output0_attn = model.gpt_neox.layers[0].attention.output[0].grad.save()
    logits = model.embed_out.output.save()
    dot_product = t.dot(logits.flatten(), tensor.flatten())
    dot_product.backward()
    
# print(logits.shape)
print(output0)
print(output0_mlp)
print(output0_attn)




with model.trace("I will be out for a ") as tracer:
    output0 = model.gpt_neox.layers[0].output[0].grad.save()
    output0_mlp = model.gpt_neox.layers[0].mlp.output.grad.save()
    output0_attn = model.gpt_neox.layers[0].attention.output[0].grad.save()
    logits = model.output.logits()
    model.output.logits.sum().backward()
    
normed_out0 = t.norm(output0, dim = -1).squeeze(0)
normed_out0_mlp = t.norm(output0_mlp, dim = -1).squeeze(0)
normed_out0_attn = t.norm(output0_attn, dim = -1).squeeze(0)
    
print(normed_out0)
print(normed_out0_mlp)
print(normed_out0_attn)


gpt2_model = LanguageModel("openai-community/gpt2", device_map = "cpu")

with gpt2_model.trace("I will be out for a ") as tracer:
    gpt2_output0 = gpt2_model.transformer.h[0].output[0].grad.save()
    gpt2_output0_mlp = gpt2_model.transformer.h[0].mlp.output.grad.save()
    gpt2_output0_attn = gpt2_model.transformer.h[0].attn.output[0].grad.save()
    gpt2_model.output.logits.sum().backward()

normed_out0_gpt2 = t.norm(gpt2_output0, dim = -1).squeeze(0)
normed_out0_mlp_gpt2 = t.norm(gpt2_output0_mlp, dim = -1).squeeze(0)
normed_out0_attn_gpt2 = t.norm(gpt2_output0_attn, dim = -1).squeeze(0)

print();print()
print(normed_out0_gpt2)
print(normed_out0_mlp_gpt2)
print(normed_out0_attn_gpt2)
'''