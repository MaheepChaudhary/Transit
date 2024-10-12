from nnsight import LanguageModel

model = LanguageModel("EleutherAI/pythia-70m", device_map = "cpu")

input_text = 'One day, a dog was roaming'

tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m')

# inputs = tokenizer(input_text, return_tensors="pt", padding='max_length', max_length=128, truncation=True)
print(model)

with model.trace(input_text) as tracer:
    output = model.embed_out.output.save()

print(output)