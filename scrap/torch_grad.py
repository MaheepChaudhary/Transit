from scrap.imports import *
from dataset import inspect_data



if __name__ == "__main__":
    
    batch_size = 16
    pythia_layers = 6
    
    train_data, val_data = inspect_data(data)
    with open(f"data/pythia_val_data_b16.pkl", "rb") as f:
        val_dataloader = pickle.load(f)

    # Load GPT-2 model and tokenizer
    model_name = 'EleutherAI/pythia-70m'
    # model_name = "openai-community/gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "mps")
    tokenizer = AutoTokenizer.from_pretrained(model_name)


    model.train()
    tokenizer.pad_token = tokenizer.eos_token
    

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
    
    device = t.device("mps")
    
    
    sent_per_layer = {
        "layer 0": [],
        "layer 1": [],
        "layer 2": [],
        "layer 3": [],
        "layer 4": [],
        "layer 5": []
        }
    
    
    for index, batch in enumerate(tqdm(val_dataloader)):

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
        
        model.zero_grad()

        inputs = {key: t.tensor(val).to(device) for key, val in batch.items()}
        
        outputs = model(**inputs)
        logits = outputs.logits  
        # loss = logits.sum()
        print(logits.shape)
        
        final_token_grads = []
        
        for sent in range(batch_size):

            layer_for_each_token = {
                "layer 0": [],
                "layer 1": [],
                "layer 2": [],
                "layer 3": [],
                "layer 4": [],
                "layer 5": []
                }
            
            for token_idx in range(128):

                
                
                token_loss = logits[sent, token_idx, :].sum()
                token_loss.backward(retain_graph = True)
                

                
                for layer_idx in range(pythia_layers):
                    layer = model.gpt_neox.layers[layer_idx].mlp.dense_4h_to_h  
                    param_grad = layer.weight.grad.clone().view(-1) # Shape: (params,)
                    layer_for_each_token[f"layer {layer_idx}"].append(t.norm(param_grad).cpu())  # Shape: (#oftoken, params)
            
            # print(np.array(layer_for_each_token["layer 0"]).shape)
            
            for layer in range(pythia_layers):
                sent_per_layer[f"layer {layer}"].append(layer_for_each_token[f"layer {layer}"])

            # print(np.array(sent_per_layer["layer 0"]).shape)
        
        
        # if index == 1:
        #     break
    
    data = []
    
    for layer_idx in range(6):
        data.append(np.log(np.mean(np.array(sent_per_layer[f"layer {layer_idx}"]), axis = 0)))
    
    with open("pythia_full_data_single_token_grad.pkl", "wb") as f:
        pickle.dump(data, f)

    # Visualize the gradients using a heatmap
    plt.figure(figsize=(15, 6))
    sns.heatmap(data, cmap='viridis', cbar=True, yticklabels=range(6), xticklabels=range(128))
    plt.xlabel('Token Index')
    plt.ylabel('Layer Index')
    plt.title('Token-wise Gradient Norms Across Layers for mlp.dense_4h_to_h on log scale')
    plt.show()
    plt.close
        
    '''
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