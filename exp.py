from imports import *
from dataset import inspect_data

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

'''


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