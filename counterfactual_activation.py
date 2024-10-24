from imports import *
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


def average_activation_for_layer(model_name):
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
        file_path = f"data/{dataset}/{model_name}/activation_resid.pkl"
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                data = pickle.load(f)
                for count, act_data in enumerate(data):
                    activations[f"layer {count}"].append(act_data)

    pprint(activations)
    avg_activations = {}
    
    # Align activations using DTW and compute the average
    for layer, acts in activations.items():
        if acts:
            reference = np.array(acts[0]).flatten()  # Flatten the reference activation
            aligned_acts = []
            for act in acts:
                flattened_act = np.array(act).flatten()  # Flatten each activation before DTW
                print()
                print(f"Reference shape: {reference.shape}, Flattened act shape: {flattened_act.shape}")
                print()
                _, path = fastdtw(reference, flattened_act, dist=euclidean)
                
                # Align activations based on DTW path
                aligned_acts.append([flattened_act[i] for i, _ in path])
                
            # Compute the mean of the aligned activations
            avg_activations[layer] = np.mean(aligned_acts, axis=0)
        else:
            raise FileNotFoundError(f"No activation data found for the specified model and layer {layer}.")

    return avg_activations

# Call the function
avg_activation = average_activation_for_layer("Pythia70m")

# class act_pythia14_70_resid_post_mlp_addn:

#     def __init__(
#         self, 
#         data,
#         model,
#         model_name,
#         dataset_name):
#         print(model)
#         self.data = data
#         self.model = model
#         self.model_name = model_name
#         self.dataset_name = dataset_name
#         self.tokenizer = self.model.tokenizer
        
#         if self.dataset_name == "tinystories":
#             self.max_length = 145
#         elif self.dataset_name == "summarisation":
#             self.max_length = 340
#         elif self.dataset_name == "alpaca":
#             self.max_length = 10
    

        
#     def activation_embeds_fn(self): # So it contains 5 layers and one last layer. 
#         self.model.eval()
        
#         activation_embeds = {
#             "layer 0": [],
#             "layer 1": [],
#             "layer 2": [],
#             "layer 3": [],
#             "layer 4": [],
#             "layer 5": []
#         }
        
#         with t.no_grad():
#             for sample in tqdm(self.data):
                
#                 inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)

#                 with self.model.trace(inputs) as tracer:
#                     output0 = self.model.gpt_neox.layers[0].output[0].save()
#                     output1 = self.model.gpt_neox.layers[1].output[0].save()
#                     output2 = self.model.gpt_neox.layers[2].output[0].save()
#                     output3 = self.model.gpt_neox.layers[3].output[0].save()
#                     output4 = self.model.gpt_neox.layers[4].output[0].save()
#                     output5 = self.model.gpt_neox.layers[5].output[0].save()
                
#                 # output0.shape -> (batch_size, 128, 2048)
#                 activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
#                 activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
#                 activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
#                 activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
#                 activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
#                 activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                
                
#         return activation_embeds

        
#     def norm(self):
        
#         # Additional norm calculations for nested structures
#         # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        
#         activation_embeds = self.activation_embeds_fn()
        
#         norm_actemb = {
#             "layer 0": [],
#             "layer 1": [],
#             "layer 2": [],
#             "layer 3": [],
#             "layer 4": []
#         }
        
        
#         norm_actemb["layer 0"] = np.mean(np.array(activation_embeds["layer 0"]), axis=0)
#         norm_actemb["layer 1"] = np.mean(np.array(activation_embeds["layer 1"]), axis=0)
#         norm_actemb["layer 2"] = np.mean(np.array(activation_embeds["layer 2"]), axis=0)
#         norm_actemb["layer 3"] = np.mean(np.array(activation_embeds["layer 3"]), axis=0)
#         norm_actemb["layer 4"] = np.mean(np.array(activation_embeds["layer 4"]), axis=0)
#         norm_actemb["layer 5"] = np.mean(np.array(activation_embeds["layer 5"]), axis=0)
        
#         # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        
#         actlist = np.array([
#             np.log(np.array(norm_actemb["layer 0"])),
#             np.log(np.array(norm_actemb["layer 1"])),
#             np.log(np.array(norm_actemb["layer 2"])),
#             np.log(np.array(norm_actemb["layer 3"])),
#             np.log(np.array(norm_actemb["layer 4"])),
#             np.log(np.array(norm_actemb["layer 5"]))
#             # mean_acts["last layer"]
#             ])
        
        
#         try:
#             os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
#             os.mkdir(f"data/{self.dataset_name}/{self.model_name}")
#         except:
#             pass
        
#         output_dir = f"data/{self.dataset_name}/{self.model_name}"
#         os.makedirs(output_dir, exist_ok=True)
        
#         with open(f"data/{self.dataset_name}/{self.model_name}/activation_resid.pkl", "wb") as f:
#             pickle.dump(actlist, f)
        
#         self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_resid.png")

#     def act_normwmean_fn_ppma(self):
        
#         normwmean_actemb_ppma = {
#             "layer 0": [],
#             "layer 1": [],
#             "layer 2": [],
#             "layer 3": [],
#             "layer 4": [],
#             "layer 5": []
#         }
        
#         for sample in tqdm(self.data):
            
#             inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
            
#             with self.model.trace(inputs) as tracer:
#                 output0_ppma = self.model.gpt_neox.layers[0].output[0].save()
#                 output1_ppma = self.model.gpt_neox.layers[1].output[0].save()
#                 output2_ppma = self.model.gpt_neox.layers[2].output[0].save()
#                 output3_ppma = self.model.gpt_neox.layers[3].output[0].save()
#                 output4_ppma = self.model.gpt_neox.layers[4].output[0].save()
#                 output5_ppma = self.model.gpt_neox.layers[5].output[0].save()
            
#             output0_ppma = output0_ppma.detach().cpu()
#             output1_ppma = output1_ppma.detach().cpu()
#             output2_ppma = output2_ppma.detach().cpu()
#             output3_ppma = output3_ppma.detach().cpu()
#             output4_ppma = output4_ppma.detach().cpu()
#             output5_ppma = output5_ppma.detach().cpu()

#             output0mean_ppma = output0_ppma - t.mean(output0_ppma, dim = 0, keepdim = True)
#             output1mean_ppma = output1_ppma - t.mean(output1_ppma, dim = 0, keepdim = True)
#             output2mean_ppma = output2_ppma - t.mean(output2_ppma, dim = 0, keepdim = True)
#             output3mean_ppma = output3_ppma - t.mean(output3_ppma, dim = 0, keepdim = True)
#             output4mean_ppma = output4_ppma - t.mean(output4_ppma, dim = 0, keepdim = True)
#             output5mean_ppma = output5_ppma - t.mean(output5_ppma, dim = 0, keepdim = True)
            
#             normwmean_actemb_ppma["layer 0"].append(t.mean(t.norm(output0mean_ppma, dim = -1), dim = 0))
#             normwmean_actemb_ppma["layer 1"].append(t.mean(t.norm(output1mean_ppma, dim = -1), dim = 0))
#             normwmean_actemb_ppma["layer 2"].append(t.mean(t.norm(output2mean_ppma, dim = -1), dim = 0))
#             normwmean_actemb_ppma["layer 3"].append(t.mean(t.norm(output3mean_ppma, dim = -1), dim = 0))
#             normwmean_actemb_ppma["layer 4"].append(t.mean(t.norm(output4mean_ppma, dim = -1), dim = 0))
#             normwmean_actemb_ppma["layer 5"].append(t.mean(t.norm(output5mean_ppma, dim = -1), dim = 0))
            
#             del output0_ppma, output1_ppma, output2_ppma, output3_ppma, output4_ppma
#             gc.collect()
            
#         return normwmean_actemb_ppma

#     def act_normwmean_ppma(self):
        
#         wmean_actemb_ppma = self.act_normwmean_fn_ppma()

        
#         normwmean_actemb_ppma = {
#             "layer 0": [],
#             "layer 1": [],
#             "layer 2": [],
#             "layer 3": [],
#             "layer 4": []
#         }
        
#         print(np.array(wmean_actemb_ppma["layer 0"]).shape)
#         normwmean_actemb_ppma["layer 0"] = np.mean(np.array(wmean_actemb_ppma["layer 0"]), axis=0)
#         normwmean_actemb_ppma["layer 1"] = np.mean(np.array(wmean_actemb_ppma["layer 1"]), axis=0)
#         normwmean_actemb_ppma["layer 2"] = np.mean(np.array(wmean_actemb_ppma["layer 2"]), axis=0)
#         normwmean_actemb_ppma["layer 3"] = np.mean(np.array(wmean_actemb_ppma["layer 3"]), axis=0)
#         normwmean_actemb_ppma["layer 4"] = np.mean(np.array(wmean_actemb_ppma["layer 4"]), axis=0)
#         normwmean_actemb_ppma["layer 5"] = np.mean(np.array(wmean_actemb_ppma["layer 5"]), axis=0)

#         actlistmean_ppma = np.array([
#             np.log(np.array(normwmean_actemb_ppma["layer 0"])),
#             np.log(np.array(normwmean_actemb_ppma["layer 1"])),
#             np.log(np.array(normwmean_actemb_ppma["layer 2"])),
#             np.log(np.array(normwmean_actemb_ppma["layer 3"])),
#             np.log(np.array(normwmean_actemb_ppma["layer 4"])),
#             np.log(np.array(normwmean_actemb_ppma["layer 5"]))
#             ])
        
#         try:
#             os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
#         except:
#             pass

#         self.plotting(data=actlistmean_ppma, name = f"figures/{self.dataset_name}/{self.model_name}/activation_resid_wmean.png")

#     def plotting(self, data, name):
#         # Create the heatmap
#         fig, ax = plt.subplots(figsize=(10, 5))  # Set figure size
#         cax = ax.imshow(data, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

#         # Add color bar to indicate the scale
#         cbar = fig.colorbar(cax, ax=ax)

#         # Set labels
#         ax.set_xlabel('Tokens')
#         ax.set_ylabel('Layers')

#         # Add labels to the right side (create twin axes sharing the same y-axis)
#         ax_right = ax.twinx()  
#         ax_right.set_ylabel('Log Scale', rotation=-90, labelpad=15)

#         # Optionally, you can add titles
#         plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of residual post-mlp addn")

#         # Show the heatmap
#         plt.savefig(name)
#         plt.close()

