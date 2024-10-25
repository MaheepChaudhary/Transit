from imports import *
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def average_activation_for_layer(model_name, dataset_used, layer_type):
    
    '''
    dataset_used: int, the length of the activations needed for the model according to the dataset
                            which we are processing the activations for to reduce the common effect of the dataset. 
    '''
    
    datasets = ["tinystories", "summarisation", "alpaca"]

    if model_name == "Pythia14m" or model_name == "Pythia70m":
        model_layer = 6
        activations = {f"layer {i}": [] for i in range(model_layer)}
    elif model_name == "Pythia160m":
        model_layer = 12
        activations = {f"layer {i}": [] for i in range(model_layer)}
    elif model_name == "Pythia410m":
        model_layer = 24
        activations = {f"layer {i}": [] for i in range(model_layer)}
    elif model_name == "Pythia1b":
        model_layer = 16
        activations = {f"layer {i}": [] for i in range(model_layer)}

    # Collect activations from datasets
    for dataset in datasets:
        file_path = f"data/{dataset}/{model_name}/{layer_type}.pkl"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                print(f"Loaded {len(data)} activation entries for {dataset}")
                for count, act_data in enumerate(data):
                    if count < model_layer:
                        activations[f"layer {count}"].append(act_data)
        else:
            print(f"File not found for {dataset}: {file_path}")

    # Process activations
    avg_activations = {}
    for layer, acts in activations.items():
        if acts:
            if dataset_used == "tinystories":
                reference = np.array(acts[0]).reshape(-1, 1)  # Flatten the reference activation
            elif dataset_used == "summarisation":
                reference = np.array(acts[1]).reshape(-1, 1)
            elif dataset_used == "alpaca":
                reference = np.array(acts[2]).reshape(-1, 1)
                
            reference_len = len(reference)
            aligned_acts = []

            for act in acts:
                flattened_act = np.array(act).reshape(-1, 1)  # Flatten each activation before DTW
                _, path = fastdtw(reference, flattened_act, dist=euclidean)

                # Align activations based on DTW path, creating an interpolated activation of reference length
                aligned_act = np.zeros((reference_len, 1))
                for ref_idx, act_idx in path:
                    aligned_act[ref_idx] += flattened_act[act_idx]
                
                # Normalize the aligned activations by the number of matches to account for repetitions
                counts = np.bincount([ref_idx for ref_idx, _ in path], minlength=reference_len)
                counts[counts == 0] = 1  # Avoid division by zero
                aligned_act /= counts.reshape(-1, 1)
                
                aligned_acts.append(aligned_act.flatten())

            # Compute the mean of the aligned activations
            avg_activations[layer] = np.mean(aligned_acts, axis=0)
        else:
            print(f"No activation data found for layer {layer}")

    return avg_activations


class counterfactual_activation_1470m:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name,
        layer_type):
        print(model)
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        self.layer_type = layer_type
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
    
    def run_and_plot_counterfactual_activations(self):
        datasets = ["tinystories", "summarisation", "alpaca"]
        counterfactual_activations = {}
        
        for dataset in datasets:
            self.dataset_name = dataset
            if dataset == "tinystories":
                self.max_length = 145
            elif dataset == "summarisation":
                self.max_length = 340
            elif dataset == "alpaca":
                self.max_length = 10
            
            activation_embeds = self.activation_embeds_fn()
            norm_actemb = {f"layer {i}": np.mean(np.array(activation_embeds[f"layer {i}"]), axis=0) for i in range(6)}
            
            counterfactual_avg_activations = average_activation_for_layer(self.model_name, self.dataset_name, self.layer_type)
            
            counterfactual_subtracted_activations = {
                layer: norm_actemb[layer] - counterfactual_avg_activations[layer]
                for layer in norm_actemb.keys()
            }
            
            counterfactual_activations[dataset] = [np.linalg.norm(counterfactual_subtracted_activations[f"layer {i}"]) for i in range(6)]
        
        # Plotting
        plt.figure(figsize=(10, 6))
        for dataset, activations in counterfactual_activations.items():
            plt.plot(range(6), activations, marker='o', label=dataset)
        
        plt.xlabel('Layer')
        plt.ylabel('Norm of Counterfactual Subtracted Activations')
        plt.title('Counterfactual Activations Across Layers for Different Datasets')
        plt.legend()
        plt.grid(True)
        plt.savefig('counterfactual_activations_plot.png')
        plt.close()
        
        return counterfactual_activations

        
    def activation_embeds_fn(self): # So it contains 5 layers and one last layer. 
        self.model.eval()
        
        activation_embeds = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": [],
            "layer 5": []
        }
        
        with t.no_grad():
            for sample in tqdm(self.data):
                
                inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)

                with self.model.trace(inputs) as tracer:
                    output0 = self.model.gpt_neox.layers[0].output[0].save()
                    output1 = self.model.gpt_neox.layers[1].output[0].save()
                    output2 = self.model.gpt_neox.layers[2].output[0].save()
                    output3 = self.model.gpt_neox.layers[3].output[0].save()
                    output4 = self.model.gpt_neox.layers[4].output[0].save()
                    output5 = self.model.gpt_neox.layers[5].output[0].save()
                    
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                
        return activation_embeds

        
    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        
        activation_embeds = self.activation_embeds_fn()
        norm_actemb = {f"layer {i}": np.mean(np.array(activation_embeds[f"layer {i}"]), axis=0) for i in range(6)}
        
        
        # Get the counterfactual average activation
        counterfactual_avg_activations = average_activation_for_layer(self.model_name, self.dataset_name, self.layer_type)
        
        # Subtract the counterfactual average activation from the activations we got
        counterfactual_subtracted_activations = {
            f"layer {i}": norm_actemb[f"layer {i}"] - np.exp(counterfactual_avg_activations[f"layer {i}"])
            for i in range(6)
        }
        
        # Convert the subtracted activations to a list for saving
        for layer in range(6):
            min_value = np.min(counterfactual_subtracted_activations[f"layer {layer}"])
            print(f"Lowest value in layer {layer}: {min_value}")
            
        actlist = np.array([
            np.abs(np.array(counterfactual_subtracted_activations[f"layer {i}"]))
            for i in range(6)
        ])
        
        try:
            os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
            os.mkdir(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass

        
        
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/{self.layer_type}.png")


    def plotting(self, data, name):
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 5))  # Set figure size
        cax = ax.imshow(data, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

        # Add color bar to indicate the scale
        cbar = fig.colorbar(cax, ax=ax)

        # Set labels
        ax.set_xlabel('Tokens')
        ax.set_ylabel('Layers')

        # Add labels to the right side (create twin axes sharing the same y-axis)
        ax_right = ax.twinx()  
        ax_right.set_ylabel('Log Scale', rotation=-90, labelpad=15)

        # Optionally, you can add titles
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of {self.layer_type}")

        # Show the heatmap
        plt.show()
        # plt.savefig(name)
        # plt.close()
