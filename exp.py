from imports import *
from dataset import inspect_data



if __name__ == "__main__":
    
    
    train_data, val_data = inspect_data(data)
    with open(f"data/pythia_val_data_b16.pkl", "rb") as f:
        val_dataloader = pickle.load(f)

    # Load GPT-2 model and tokenizer
    model_name = 'EleutherAI/pythia-70m'
    # model_name = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    # Input text
    input_text = 'One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt. Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt." Together, they shared the needle and sewed the button on Lilys shirt. It was not difficult for them because they were sharing and helping each other. After they finished, Lily thanked her mom for sharing the needle and fixing her shirt. They both felt happy because they had shared and worked together.'
    # inputs = tokenizer(input_text, return_tensors="pt")
    
    # mlp_gradients = None
    # attn_gradients = None

    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    
    '''
    inputs_text = tokenizer(
    input_text,
    padding='max_length',  # Pad to max length if necessary
    truncation=True,       # Truncate if input is longer than the model's max length
    max_length=32,         # Set max length for tokenized input
    return_tensors='pt'    # Return PyTorch tensors (can also use 'tf' for TensorFlow)
    )

    def capture_mlp_output(module, input, output):
        global mlp_gradients
        mlp_gradients = output[0]  # Capture the output for later use

    def capture_attn_output(module, input, output):
        global attn_gradients
        attn_gradients = output[0] 
        
    # model.gpt_neox.layers[0].mlp.register_full_backward_hook(capture_mlp_output)
    # model.gpt_neox.layers[0].attention.register_full_backward_hook(capture_attn_output)
    model.transformer.h[0].mlp.register_full_backward_hook(capture_mlp_output)  
    model.transformer.h[0].attn.register_full_backward_hook(capture_attn_output)

    outputs = model(**inputs_text)
    logits = outputs.logits
    loss = logits.sum()
    loss.backward()

    print(mlp_gradients)
    print(attn_gradients)
    '''
    all_data_mlp = {
        "Layer 0": [],
        "Layer 1": [],
        "Layer 2": [],
        "Layer 3": [],
        "Layer 4": [],
    }
    
    all_data_attn = {
        "Layer 0": [],
        "Layer 1": [],
        "Layer 2": [],
        "Layer 3": [],
        "Layer 4": [],
    }
    
    all_data_resid = {
        "Layer 0": [],
        "Layer 1": [],
        "Layer 2": [],
        "Layer 3": [],
        "Layer 4": [],
    }
    

    
    for i, batch in enumerate(tqdm(val_data)):

        inputs = tokenizer(
            batch['text'],
            padding='max_length',  # Pad to max length if necessary
            truncation=True,       # Truncate if input is longer than the model's max length
            max_length=128,         # Set max length for tokenized input
            return_tensors='pt'    # Return PyTorch tensors (can also use 'tf' for TensorFlow)
        )

        mlp_gradients = None
        attn_gradients = None
        mlp_gradients1 = None
        attn_gradients1 = None
        mlp_gradients2 = None
        attn_gradients2 = None
        mlp_gradients3 = None
        attn_gradients3 = None
        mlp_gradients4 = None
        attn_gradients4 = None

        def capture_mlp_output(module, input, output):
            global mlp_gradients
            mlp_gradients = output[0]  # Capture the output for later use

        def capture_attn_output(module, input, output):
            global attn_gradients
            attn_gradients = output[0]  # Capture the output for later use
            
        def capture_mlp_output1(module, input, output):
            global mlp_gradients1
            mlp_gradients1 = output[0]  # Capture the output for later use

        def capture_attn_output1(module, input, output):
            global attn_gradients1
            attn_gradients1 = output[0]  # Capture the output for later use

        def capture_mlp_output2(module, input, output):
            global mlp_gradients2
            mlp_gradients2 = output[0]  # Capture the output for later use

        def capture_attn_output2(module, input, output):
            global attn_gradients2
            attn_gradients2 = output[0]  # Capture the output for later use


        def capture_mlp_output3(module, input, output):
            global mlp_gradients3
            mlp_gradients3 = output[0]  # Capture the output for later use

        def capture_attn_output3(module, input, output):
            global attn_gradients3
            attn_gradients3 = output[0]  # Capture the output for later use
            
            
        def capture_mlp_output4(module, input, output):
            global mlp_gradients4
            mlp_gradients4 = output[0]  # Capture the output for later use

        def capture_attn_output4(module, input, output):
            global attn_gradients4
            attn_gradients4 = output[0]  # Capture the output for later use


        model.gpt_neox.layers[0].mlp.register_backward_hook(capture_mlp_output)
        model.gpt_neox.layers[0].attention.register_backward_hook(capture_attn_output)
        
        model.gpt_neox.layers[1].mlp.register_backward_hook(capture_mlp_output1)
        model.gpt_neox.layers[1].attention.register_backward_hook(capture_attn_output1)
        
        model.gpt_neox.layers[2].mlp.register_backward_hook(capture_mlp_output2)
        model.gpt_neox.layers[2].attention.register_backward_hook(capture_attn_output2)
        
        model.gpt_neox.layers[3].mlp.register_backward_hook(capture_mlp_output3)
        model.gpt_neox.layers[3].attention.register_backward_hook(capture_attn_output3)
        
        model.gpt_neox.layers[4].mlp.register_backward_hook(capture_mlp_output4)
        model.gpt_neox.layers[4].attention.register_backward_hook(capture_attn_output4)
        

        outputs = model(**inputs)
        logits = outputs.logits  
        loss = logits.sum()
        loss.backward()

        # print(mlp_gradients)
        # print(attn_gradients)

        # Combine gradients if needed
        combined_grad = mlp_gradients + attn_gradients
        combined_grad1 = mlp_gradients1 + attn_gradients1
        combined_grad2 = mlp_gradients2 + attn_gradients2
        combined_grad3 = mlp_gradients3 + attn_gradients3
        combined_grad4 = mlp_gradients4 + attn_gradients4
        
        mean_norm_mlp_grad = t.norm(mlp_gradients.squeeze(0), dim = -1)
        mean_norm_mlp_grad1 = t.norm(mlp_gradients1.squeeze(0), dim = -1)
        mean_norm_mlp_grad2 = t.norm(mlp_gradients2.squeeze(0), dim = -1)
        mean_norm_mlp_grad3 = t.norm(mlp_gradients3.squeeze(0), dim = -1)
        mean_norm_mlp_grad4 = t.norm(mlp_gradients4.squeeze(0), dim = -1)
        
        
        mean_norm_attn_grad = t.norm(attn_gradients.squeeze(0), dim = -1)
        mean_norm_attn_grad1 = t.norm(attn_gradients1.squeeze(0), dim = -1)
        mean_norm_attn_grad2 = t.norm(attn_gradients2.squeeze(0), dim = -1)
        mean_norm_attn_grad3 = t.norm(attn_gradients3.squeeze(0), dim = -1)
        mean_norm_attn_grad4 = t.norm(attn_gradients4.squeeze(0), dim = -1)


        mean_norm_combined_grad = t.norm(combined_grad.squeeze(0), dim = -1)
        mean_norm_combined_grad1 = t.norm(combined_grad1.squeeze(0), dim = -1)
        mean_norm_combined_grad2 = t.norm(combined_grad2.squeeze(0), dim = -1)
        mean_norm_combined_grad3 = t.norm(combined_grad3.squeeze(0), dim = -1)
        mean_norm_combined_grad4 = t.norm(combined_grad4.squeeze(0), dim = -1)
        
        
        all_data_mlp["Layer 0"].append(mean_norm_mlp_grad)
        all_data_mlp["Layer 1"].append(mean_norm_mlp_grad1)
        all_data_mlp["Layer 2"].append(mean_norm_mlp_grad2)
        all_data_mlp["Layer 3"].append(mean_norm_mlp_grad3)
        all_data_mlp["Layer 4"].append(mean_norm_mlp_grad4)
        
        all_data_attn["Layer 0"].append(mean_norm_attn_grad)
        all_data_attn["Layer 1"].append(mean_norm_attn_grad1)
        all_data_attn["Layer 2"].append(mean_norm_attn_grad2)
        all_data_attn["Layer 3"].append(mean_norm_attn_grad3)
        all_data_attn["Layer 4"].append(mean_norm_attn_grad4)
        
        all_data_resid["Layer 0"].append(mean_norm_combined_grad)
        all_data_resid["Layer 1"].append(mean_norm_combined_grad1)
        all_data_resid["Layer 2"].append(mean_norm_combined_grad2)
        all_data_resid["Layer 3"].append(mean_norm_combined_grad3)
        all_data_resid["Layer 4"].append(mean_norm_combined_grad4)
        
        # if i == 1:
        #     break

    with open("data/all_data_mlp.pkl", "wb") as f:
        pickle.dump(all_data_mlp, f)
        
    with open("data/all_data_attn.pkl", "wb") as f1:
        pickle.dump(all_data_attn, f1)
        
    with open("data/all_data_resid.pkl", "wb") as f2:
        pickle.dump(all_data_resid, f2)
    
    
    def plot():
        with open("data/all_data_mlp.pkl", "rb") as f:
            mlp = pickle.load(f)
            
        with open("data/all_data_attn.pkl", "rb") as f1:
            attn = pickle.load(f1)
            
        with open("data/all_data_resid.pkl", "rb") as f2:
            resid = pickle.load(f2)
        
        data_mlp = np.array([v for k, v in sorted(mlp.items())])
        data_attn = np.array([v for k, v in sorted(attn.items())])
        data_resid = np.array([v for k, v in sorted(resid.items())])
        
        data_mlp = np.mean(data_mlp, axis = 1)
        data_attn = np.mean(data_attn, axis = 1)
        data_resid = np.mean(data_resid, axis = 1)
        
        print(data_mlp)
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
        cax = ax.imshow(data_mlp, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

        # Add color bar to indicate the scale
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Gradient Magnitude')

        # Set labels
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Layers')

        # Set x-ticks and y-ticks
        ax.set_xticks(np.arange(data_mlp.shape[1]))
        ax.set_yticks(np.arange(data_mlp.shape[0]))
        ax.set_xticklabels([f'Token {i}' for i in range(data_mlp.shape[1])])
        ax.set_yticklabels(list(sorted(mlp.keys())))

        # Optionally, you can add titles
        plt.title(f"[Pythia]: Gradient of Pythia MLP Output")

        # Save the plot
        plt.savefig(f"mfigures/MLP.png")
        plt.close()
        
        
        fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
        cax = ax.imshow(data_attn, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

        # Add color bar to indicate the scale
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Gradient Magnitude')

        # Set labels
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Layers')

        # Set x-ticks and y-ticks
        ax.set_xticks(np.arange(data_attn.shape[1]))
        ax.set_yticks(np.arange(data_attn.shape[0]))
        ax.set_xticklabels([f'Token {i}' for i in range(data_attn.shape[1])])
        ax.set_yticklabels(list(sorted(attn.keys())))

        # Optionally, you can add titles
        plt.title(f"[Pythia]: Gradient of Pythia Attention Output")

        # Save the plot
        plt.savefig(f"mfigures/Attention.png")
        plt.close()
        
        
        fig, ax = plt.subplots(figsize=(12, 6))  # Set figure size
        cax = ax.imshow(data_resid, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

        # Add color bar to indicate the scale
        cbar = fig.colorbar(cax, ax=ax)
        cbar.set_label('Gradient Magnitude')

        # Set labels
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Layers')

        # Set x-ticks and y-ticks
        ax.set_xticks(np.arange(data_resid.shape[1]))
        ax.set_yticks(np.arange(data_resid.shape[0]))
        ax.set_xticklabels([f'Token {i}' for i in range(data_resid.shape[1])])
        ax.set_yticklabels(list(sorted(resid.keys())))

        # Optionally, you can add titles
        plt.title(f"[Pythia]: Gradient of Pythia Resid Output")

        # Save the plot
        plt.savefig(f"mfigures/Resid.png")
        plt.close()
        
        
    # grads(val_data, model, tokenizer)
    plot()

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