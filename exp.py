from imports import *
from dataset import inspect_data

model = LanguageModel("EleutherAI/pythia-70m", device_map="cpu")
train_data, val_data = inspect_data(data)

random_samples = [{"text":"Hello world. How are you?"} for i in range(20)]
for index, sample in enumerate(random_samples):
    pprint(sample['text'])
    print()
    with model.trace(sample['text']) as tracer:
        output0 = model.gpt_neox.layers[0].output[0].grad.save()
        output1 = model.gpt_neox.layers[1].output[0].grad.save()
        output2 = model.gpt_neox.layers[2].output[0].grad.save()
        output3 = model.gpt_neox.layers[3].output[0].grad.save()
        output4 = model.gpt_neox.layers[4].output[0].grad.save()
        model.output.logits.sum().backward()
        
print(output0)
print(output1)
print(output2)
print(output3)
print(output4)