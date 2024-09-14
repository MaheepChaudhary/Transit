from imports import *
from dataset import inspect_data



if __name__ == "__main__":
    
    train_data, val_data = inspect_data(data)
    with open(f"data/pythia_val_data_b16.pkl", "rb") as f:
        val_dataloader = pickle.load(f)

    # Load GPT-2 model and tokenizer
    model_name = 'EleutherAI/pythia-70m'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Input text
    input_text = 'One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt. Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt." Together, they shared the needle and sewed the button on Lilys shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together.'
    # inputs = tokenizer(input_text, return_tensors="pt")

    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    for batch in val_data:

        inputs = tokenizer(
            batch['text'],
            padding='max_length',  # Pad to max length if necessary
            truncation=True,       # Truncate if input is longer than the model's max length
            max_length=32,         # Set max length for tokenized input
            return_tensors='pt'    # Return PyTorch tensors (can also use 'tf' for TensorFlow)
        )

        mlp_gradients = None
        attn_gradients = None

        def capture_mlp_output(module, input, output):
            global mlp_gradients
            mlp_gradients = output[0]  # Capture the output for later use

        def capture_attn_output(module, input, output):
            global attn_gradients
            attn_gradients = output[0]  # Capture the output for later use
            
        def capture_mlp_output1(module, input, output):
            global mlp_gradients
            mlp_gradients1 = output[0]  # Capture the output for later use

        def capture_attn_output1(module, input, output):
            global attn_gradients
            attn_gradients1 = output[0]  # Capture the output for later use

        def capture_mlp_output2(module, input, output):
            global mlp_gradients
            mlp_gradients2 = output[0]  # Capture the output for later use

        def capture_attn_output2(module, input, output):
            global attn_gradients
            attn_gradients2 = output[0]  # Capture the output for later use


        def capture_mlp_output3(module, input, output):
            global mlp_gradients
            mlp_gradients3 = output[0]  # Capture the output for later use

        def capture_attn_output3(module, input, output):
            global attn_gradients
            attn_gradients3 = output[0]  # Capture the output for later use
            
            
        def capture_mlp_output4(module, input, output):
            global mlp_gradients
            mlp_gradients4 = output[0]  # Capture the output for later use

        def capture_attn_output4(module, input, output):
            global attn_gradients
            attn_gradients4 = output[0]  # Capture the output for later use


        model.gpt_neox.layers[0].mlp.register_backward_hook(capture_mlp_output)
        model.gpt_neox.layers[0].attention.register_backward_hook(capture_attn_output)
        
        model.gpt_neox.layers[0].mlp.register_backward_hook(capture_mlp_output1)
        model.gpt_neox.layers[0].attention.register_backward_hook(capture_attn_output1)
        
        model.gpt_neox.layers[0].mlp.register_backward_hook(capture_mlp_output2)
        model.gpt_neox.layers[0].attention.register_backward_hook(capture_attn_output2)
        
        model.gpt_neox.layers[0].mlp.register_backward_hook(capture_mlp_output3)
        model.gpt_neox.layers[0].attention.register_backward_hook(capture_attn_output3)
        
        model.gpt_neox.layers[0].mlp.register_backward_hook(capture_mlp_output4)
        model.gpt_neox.layers[0].attention.register_backward_hook(capture_attn_output4)
        

        outputs = model(**inputs)
        logits = outputs.logits  
        loss = logits.sum()
        loss.backward()

        # Combine gradients if needed
        combined_grad = mlp_gradients + attn_gradients
        combined_grad1 = 

        print("MLP Output Gradient (First Block):", mlp_gradients)
        print("Attention Output Gradient (First Block):", attn_gradients)
        print("Combined Gradient (MLP + Attention in First Block):", combined_grad)
        


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