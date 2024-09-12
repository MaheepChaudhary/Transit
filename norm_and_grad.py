from imports import *


class act_pythia_resid_post_mlp_addn:

    def __init__(
        self, 
        title, 
        name,
        model,
        dataloader):
        
        self.title = title
        self.name = name
        self.model = model
        self.dataloader = dataloader
        
        
        try:
            with open(f"data/pythia_activation_embeds_{self.name}.pkl", "rb") as f:
                self.activation_embeds = pickle.load(f)
        except:
            print("Generating pickle file of activation embeds")
            self.activation_embeds = self.activation_embeds_fn()
        
    def activation_embeds_fn(self): # So it contains 5 layers and one last layer. 
        self.model.eval()
        
        activation_embeds = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        with t.no_grad():
            for batch in tqdm(self.dataloader):
                
                with self.model.trace(batch["input_ids"]) as tracer:
                    output0 = self.model.gpt_neox.layers[0].output[0].save()
                    output1 = self.model.gpt_neox.layers[1].output[0].save()
                    output2 = self.model.gpt_neox.layers[2].output[0].save()
                    output3 = self.model.gpt_neox.layers[3].output[0].save()
                    output4 = self.model.gpt_neox.layers[4].output[0].save()

                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
                activation_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
                activation_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
                activation_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
                activation_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
                
        with open(f"data/pythia_activation_embeds_{self.name}.pkl", "wb") as f:
            pickle.dump(activation_embeds, f)
                
        return activation_embeds

        
    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        norm_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        
        norm_actemb["layer 0"] = np.mean(self.activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(self.activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(self.activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(self.activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(self.activation_embeds["layer 4"], axis=0)
        
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        actlist = np.array([
            np.log(np.array(norm_actemb["layer 0"])),
            np.log(np.array(norm_actemb["layer 1"])),
            np.log(np.array(norm_actemb["layer 2"])),
            np.log(np.array(norm_actemb["layer 3"])),
            np.log(np.array(norm_actemb["layer 4"])),
            # mean_acts["last layer"]
            ])
        
        print(actlist)
        self.plotting(data=actlist, name = f"figures/pythia_activation_embeds_{self.name}.png")


    # def normwmean(self, data):
        
    #     # Additional norm calculations for nested structures
    #     # assert np.array(self.actemb["layer 0"]).shape[1] == 128
    #     normwmean_actemb = {
    #         "layer 0": [],
    #         "layer 1": [],
    #         "layer 2": [],
    #         "layer 3": [],
    #         "layer 4": []
    #     }
        
    #     normwmean_actemb_mod = {
    #         "layer 0": [],
    #         "layer 1": [],
    #         "layer 2": [],
    #         "layer 3": [],
    #         "layer 4": []
    #     }
        
    #     normwmean_actemb["layer 0"] = np.mean(data["layer 0"], axis=0).squeeze(0)
    #     normwmean_actemb["layer 1"] = np.mean(data["layer 1"], axis=0).squeeze(0)
    #     normwmean_actemb["layer 2"] = np.mean(data["layer 2"], axis=0).squeeze(0)
    #     normwmean_actemb["layer 3"] = np.mean(data["layer 3"], axis=0).squeeze(0)
    #     normwmean_actemb["layer 4"] = np.mean(data["layer 4"], axis=0).squeeze(0)
        
    #     normwmean_actemb_mod["layer 0"] = np.array(normwmean_actemb["layer 0"]) - np.mean(np.array(normwmean_actemb["layer 0"]), axis = 0)
    #     normwmean_actemb_mod["layer 1"] = np.array(normwmean_actemb["layer 1"]) - np.mean(np.array(normwmean_actemb["layer 1"]), axis = 0)
    #     normwmean_actemb_mod["layer 2"] = np.array(normwmean_actemb["layer 2"]) - np.mean(np.array(normwmean_actemb["layer 2"]), axis = 0)
    #     normwmean_actemb_mod["layer 3"] = np.array(normwmean_actemb["layer 3"]) - np.mean(np.array(normwmean_actemb["layer 3"]), axis = 0)
    #     normwmean_actemb_mod["layer 4"] = np.array(normwmean_actemb["layer 4"]) - np.mean(np.array(normwmean_actemb["layer 4"]), axis = 0)
        
        
    #     actlistmean = np.array([
    #         np.log(np.array(normwmean_actemb["layer 0"])),
    #         np.log(np.array(normwmean_actemb["layer 1"])),
    #         np.log(np.array(normwmean_actemb["layer 2"])),
    #         np.log(np.array(normwmean_actemb["layer 3"])),
    #         np.log(np.array(normwmean_actemb["layer 4"])),
    #         # mean_acts["last layer"]
    #         ])

    #     self.plotting(data=actlistmean, name = f"mfigures/activation_normwmean_{self.name}.png")

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
        plt.title(f"[Pythia]:Activation of {self.title}")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class grad_pythia_resid_post_mlp_addn:
    
    def __init__(
        self, 
        model, 
        dataloader, 
        title, 
        name
        ):
        
        self.model = model
        self.dataloader = dataloader
        self.title = title
        self.name = name
        
        try:
            with open(f"data/pythia_grad_norm_{name}.pkl", "rb") as f:
                grads = pickle.load(f)    
            self.grads = grads  
        except:
            print("Generating pickle file of gradient embeds")
            self.grads = self.get_grads()
            
    
    
    def get_grads(self):
        
        grad_embeds = {
        "layer 0": [],
        "layer 1": [],
        "layer 2": [],
        "layer 3": [],
        "layer 4": []
        }
        
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
            
                output0 = self.model.gpt_neox.layers[0].output[0].grad.save()
                output1 = self.model.gpt_neox.layers[1].output[0].grad.save()
                output2 = self.model.gpt_neox.layers[2].output[0].grad.save()
                output3 = self.model.gpt_neox.layers[3].output[0].grad.save()
                output4 = self.model.gpt_neox.layers[4].output[0].grad.save()
                
                self.model.output.logits.sum().backward()
            
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            grad_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
            grad_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
            grad_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
            grad_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
            grad_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
            
        with open(f"data/pythia_grad_norm_{self.name}.pkl", "wb") as f:
            pickle.dump(grad_embeds, f)
            
        return grad_embeds

    def grad_norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        grad_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        
        grad_actemb["layer 0"] = np.mean(self.grads["layer 0"], axis=0)
        grad_actemb["layer 1"] = np.mean(self.grads["layer 1"], axis=0)
        grad_actemb["layer 2"] = np.mean(self.grads["layer 2"], axis=0)
        grad_actemb["layer 3"] = np.mean(self.grads["layer 3"], axis=0)
        grad_actemb["layer 4"] = np.mean(self.grads["layer 4"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        gradlist = np.array([
            np.log(np.array(grad_actemb["layer 0"])),
            np.log(np.array(grad_actemb["layer 1"])),
            np.log(np.array(grad_actemb["layer 2"])),
            np.log(np.array(grad_actemb["layer 3"])),
            np.log(np.array(grad_actemb["layer 4"])),
            # mean_acts["last layer"]
            ])
        print(gradlist)
        self.plotting(data=gradlist, name = f"figures/pythia_grad_embed_{self.name}.png")
    
    # def normwmean(self):
        
    #     # Additional norm calculations for nested structures
    #     # assert np.array(self.actemb["layer 0"]).shape[1] == 128
    #     normwmean_grad = {
    #         "layer 0": [],
    #         "layer 1": [],
    #         "layer 2": [],
    #         "layer 3": [],
    #         "layer 4": []
    #     }
        
    #     normwmean_grad_mod = {
    #         "layer 0": [],
    #         "layer 1": [],
    #         "layer 2": [],
    #         "layer 3": [],
    #         "layer 4": []
    #     }
        
    #     normwmean_grad["layer 0"] = np.mean(self.grads["layer 0"], axis=0)
    #     normwmean_grad["layer 1"] = np.mean(self.grads["layer 1"], axis=0)
    #     normwmean_grad["layer 2"] = np.mean(self.grads["layer 2"], axis=0)
    #     normwmean_grad["layer 3"] = np.mean(self.grads["layer 3"], axis=0)
    #     normwmean_grad["layer 4"] = np.mean(self.grads["layer 4"], axis=0)
    
    #     normwmean_grad_mod["layer 0"] = np.array(normwmean_grad["layer 0"]) - np.mean(np.array(normwmean_grad["layer 0"]), axis = 0)
    #     normwmean_grad_mod["layer 1"] = np.array(normwmean_grad["layer 1"]) - np.mean(np.array(normwmean_grad["layer 1"]), axis = 0)
    #     normwmean_grad_mod["layer 2"] = np.array(normwmean_grad["layer 2"]) - np.mean(np.array(normwmean_grad["layer 2"]), axis = 0)
    #     normwmean_grad_mod["layer 3"] = np.array(normwmean_grad["layer 3"]) - np.mean(np.array(normwmean_grad["layer 3"]), axis = 0)
    #     normwmean_grad_mod["layer 4"] = np.array(normwmean_grad["layer 4"]) - np.mean(np.array(normwmean_grad["layer 4"]), axis = 0)
        
    #     # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        
    #     gradlistmean = np.array([
    #         np.log(np.array(normwmean_grad_mod["layer 0"])),
    #         np.log(np.array(normwmean_grad_mod["layer 1"])),
    #         np.log(np.array(normwmean_grad_mod["layer 2"])),
    #         np.log(np.array(normwmean_grad_mod["layer 3"])),
    #         np.log(np.array(normwmean_grad_mod["layer 4"])),
    #         # mean_acts["last layer"]
    #         ])
        
    #     self.plotting(data=gradlistmean, name = "mfigures/grad_layer_seq_normwmean_grad_post_mlp_addn_resid.png")

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
        plt.title(f"[Pythia]:Gradient of {self.title}")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia_mlp:

    def __init__(
        self, 
        title, 
        name,
        model,
        dataloader):
        
        self.title = title
        self.name = name
        self.model = model
        self.dataloader = dataloader
        
        
        try:
            with open(f"data/pythia_activation_embeds_{self.name}.pkl", "rb") as f:
                self.activation_embeds = pickle.load(f)
        except:
            print("Generating pickle file of activation embeds")
            self.activation_embeds = self.activation_embeds_fn()
        
    def activation_embeds_fn(self): # So it contains 5 layers and one last layer. 
        self.model.eval()
        
        activation_embeds = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        with t.no_grad():
            for batch in tqdm(self.dataloader):
                
                with self.model.trace(batch["input_ids"]) as tracer:
                    output0 = self.model.gpt_neox.layers[0].mlp.output.save()
                    output1 = self.model.gpt_neox.layers[1].mlp.output.save()
                    output2 = self.model.gpt_neox.layers[2].mlp.output.save()
                    output3 = self.model.gpt_neox.layers[3].mlp.output.save()
                    output4 = self.model.gpt_neox.layers[4].mlp.output.save()

                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
                activation_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
                activation_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
                activation_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
                activation_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
                
        with open(f"data/pythia_activation_embeds_{self.name}.pkl", "wb") as f:
            pickle.dump(activation_embeds, f)
                
        return activation_embeds

        
    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        norm_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        
        norm_actemb["layer 0"] = np.mean(self.activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(self.activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(self.activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(self.activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(self.activation_embeds["layer 4"], axis=0)
        
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        actlist = np.array([
            np.log(np.array(norm_actemb["layer 0"])),
            np.log(np.array(norm_actemb["layer 1"])),
            np.log(np.array(norm_actemb["layer 2"])),
            np.log(np.array(norm_actemb["layer 3"])),
            np.log(np.array(norm_actemb["layer 4"])),
            # mean_acts["last layer"]
            ])
        
        print(actlist)
        self.plotting(data=actlist, name = f"figures/pythia_activation_embeds_{self.name}.png")



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
        plt.title(f"[Pythia]:Activation of {self.title}")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class grad_pythia_mlp:
    
    def __init__(
        self, 
        model, 
        dataloader, 
        title, 
        name
        ):
        
        self.model = model
        self.dataloader = dataloader
        self.title = title
        self.name = name
        
        try:
            with open(f"data/pythia_grad_norm_{name}.pkl", "rb") as f:
                grads = pickle.load(f)    
            self.grads = grads  
        except:
            print("Generating pickle file of gradient embeds")
            self.grads = self.get_grads()
            
    
    
    def get_grads(self):
        
        grad_embeds = {
        "layer 0": [],
        "layer 1": [],
        "layer 2": [],
        "layer 3": [],
        "layer 4": []
        }
        
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
            
                output0 = self.model.gpt_neox.layers[0].mlp.output.grad.save()
                output1 = self.model.gpt_neox.layers[1].mlp.output.grad.save()
                output2 = self.model.gpt_neox.layers[2].mlp.output.grad.save()
                output3 = self.model.gpt_neox.layers[3].mlp.output.grad.save()
                output4 = self.model.gpt_neox.layers[4].mlp.output.grad.save()
                
                self.model.output.logits.sum().backward()
            
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            grad_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
            grad_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
            grad_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
            grad_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
            grad_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
            
        with open(f"data/pythia_grad_norm_{self.name}.pkl", "wb") as f:
            pickle.dump(grad_embeds, f)
            
        return grad_embeds

    def grad_norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        grad_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        
        grad_actemb["layer 0"] = np.mean(self.grads["layer 0"], axis=0)
        grad_actemb["layer 1"] = np.mean(self.grads["layer 1"], axis=0)
        grad_actemb["layer 2"] = np.mean(self.grads["layer 2"], axis=0)
        grad_actemb["layer 3"] = np.mean(self.grads["layer 3"], axis=0)
        grad_actemb["layer 4"] = np.mean(self.grads["layer 4"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        gradlist = np.array([
            np.log(np.array(grad_actemb["layer 0"])),
            np.log(np.array(grad_actemb["layer 1"])),
            np.log(np.array(grad_actemb["layer 2"])),
            np.log(np.array(grad_actemb["layer 3"])),
            np.log(np.array(grad_actemb["layer 4"])),
            # mean_acts["last layer"]
            ])
        print(gradlist)
        self.plotting(data=gradlist, name = f"figures/pythia_grad_embed_{self.name}.png")
    
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
        plt.title(f"[Pythia]:Gradient of {self.title}")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia_attention:

    def __init__(
        self, 
        title, 
        name,
        model,
        dataloader):
        
        self.title = title
        self.name = name
        self.model = model
        self.dataloader = dataloader
        
        
        try:
            with open(f"data/pythia_activation_embeds_{self.name}.pkl", "rb") as f:
                self.activation_embeds = pickle.load(f)
        except:
            print("Generating pickle file of activation embeds")
            self.activation_embeds = self.activation_embeds_fn()
        
    def activation_embeds_fn(self): # So it contains 5 layers and one last layer. 
        self.model.eval()
        
        activation_embeds = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        with t.no_grad():
            for batch in tqdm(self.dataloader):
                
                with self.model.trace(batch["input_ids"]) as tracer:
                    output0 = self.model.gpt_neox.layers[0].attention.output[0].save()
                    output1 = self.model.gpt_neox.layers[1].attention.output[0].save()
                    output2 = self.model.gpt_neox.layers[2].attention.output[0].save()
                    output3 = self.model.gpt_neox.layers[3].attention.output[0].save()
                    output4 = self.model.gpt_neox.layers[4].attention.output[0].save()

                # output0.shape -> (batch_size, 128, 2048)
                print(output0.shape)
                activation_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
                activation_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
                activation_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
                activation_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
                activation_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
                
        with open(f"data/pythia_activation_embeds_{self.name}.pkl", "wb") as f:
            pickle.dump(activation_embeds, f)
                
        return activation_embeds

        
    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        norm_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        
        norm_actemb["layer 0"] = np.mean(self.activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(self.activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(self.activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(self.activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(self.activation_embeds["layer 4"], axis=0)
        
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        actlist = np.array([
            np.log(np.array(norm_actemb["layer 0"])),
            np.log(np.array(norm_actemb["layer 1"])),
            np.log(np.array(norm_actemb["layer 2"])),
            np.log(np.array(norm_actemb["layer 3"])),
            np.log(np.array(norm_actemb["layer 4"])),
            # mean_acts["last layer"]
            ])
        
        print(actlist)
        self.plotting(data=actlist, name = f"figures/pythia_activation_embeds_{self.name}.png")



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
        plt.title(f"[Pythia]:Activation of {self.title}")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class grad_pythia_attention:
    
    def __init__(
        self, 
        model, 
        dataloader, 
        title, 
        name
        ):
        
        self.model = model
        self.dataloader = dataloader
        self.title = title
        self.name = name
        
        try:
            with open(f"data/pythia_grad_norm_{name}.pkl", "rb") as f:
                grads = pickle.load(f)    
            self.grads = grads  
        except:
            print("Generating pickle file of gradient embeds")
            self.grads = self.get_grads()
            
    
    
    def get_grads(self):
        
        grad_embeds = {
        "layer 0": [],
        "layer 1": [],
        "layer 2": [],
        "layer 3": [],
        "layer 4": []
        }
        
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
            
                output0 = self.model.gpt_neox.layers[0].attention.output[0].grad.save()
                output1 = self.model.gpt_neox.layers[1].attention.output[0].grad.save()
                output2 = self.model.gpt_neox.layers[2].attention.output[0].grad.save()
                output3 = self.model.gpt_neox.layers[3].attention.output[0].grad.save()
                output4 = self.model.gpt_neox.layers[4].attention.output[0].grad.save()
                
                self.model.output.logits.sum().backward()
            
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            grad_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
            grad_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
            grad_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
            grad_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
            grad_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
            
        with open(f"data/pythia_grad_norm_{self.name}.pkl", "wb") as f:
            pickle.dump(grad_embeds, f)
            
        return grad_embeds

    def grad_norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        grad_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        
        grad_actemb["layer 0"] = np.mean(self.grads["layer 0"], axis=0)
        grad_actemb["layer 1"] = np.mean(self.grads["layer 1"], axis=0)
        grad_actemb["layer 2"] = np.mean(self.grads["layer 2"], axis=0)
        grad_actemb["layer 3"] = np.mean(self.grads["layer 3"], axis=0)
        grad_actemb["layer 4"] = np.mean(self.grads["layer 4"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        gradlist = np.array([
            np.log(np.array(grad_actemb["layer 0"])),
            np.log(np.array(grad_actemb["layer 1"])),
            np.log(np.array(grad_actemb["layer 2"])),
            np.log(np.array(grad_actemb["layer 3"])),
            np.log(np.array(grad_actemb["layer 4"])),
            # mean_acts["last layer"]
            ])
        print(gradlist)
        self.plotting(data=gradlist, name = f"figures/pythia_grad_embed_{self.name}.png")


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
        plt.title(f"[Pythia]:Gradient of {self.title}")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_gpt2_resid_post_mlp_addn:

    def __init__(
        self, 
        title, 
        name,
        model,
        dataloader):
        
        self.title = title
        self.name = name
        self.model = model
        self.dataloader = dataloader
        
        
        try:
            with open(f"data/gpt2_activation_embeds_{self.name}.pkl", "rb") as f:
                self.activation_embeds = pickle.load(f)
        except:
            print("Generating pickle file of activation embeds")
            self.activation_embeds = self.activation_embeds_fn()
        
    def activation_embeds_fn(self): # So it contains 5 layers and one last layer. 
        self.model.eval()
        
        activation_embeds = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": [],
            "layer 5": [],
            "layer 6": [],
            "layer 7": [],
            "layer 8": [],
            "layer 9": [],
            "layer 10": [],
            "layer 11": []
        }
        
        with t.no_grad():
            for batch in tqdm(self.dataloader):
                
                with self.model.trace(batch["input_ids"]) as tracer:
                    output0 = self.model.transformer.h[0].output[0].save()
                    output1 = self.model.transformer.h[1].output[0].save()
                    output2 = self.model.transformer.h[2].output[0].save()
                    output3 = self.model.transformer.h[3].output[0].save()
                    output4 = self.model.transformer.h[4].output[0].save()
                    output5 = self.model.transformer.h[5].output[0].save()
                    output6 = self.model.transformer.h[6].output[0].save()
                    output7 = self.model.transformer.h[7].output[0].save()
                    output8 = self.model.transformer.h[8].output[0].save()
                    output9 = self.model.transformer.h[9].output[0].save()
                    output10 = self.model.transformer.h[10].output[0].save()
                    output11 = self.model.transformer.h[11].output[0].save()

                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
                activation_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
                activation_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
                activation_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
                activation_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
                activation_embeds["layer 5"].append(t.mean(t.norm(output5, dim = -1), dim = 0))
                activation_embeds["layer 6"].append(t.mean(t.norm(output6, dim = -1), dim = 0))
                activation_embeds["layer 7"].append(t.mean(t.norm(output7, dim = -1), dim = 0))
                activation_embeds["layer 8"].append(t.mean(t.norm(output8, dim = -1), dim = 0))
                activation_embeds["layer 9"].append(t.mean(t.norm(output9, dim = -1), dim = 0))
                activation_embeds["layer 10"].append(t.mean(t.norm(output10, dim = -1), dim = 0))
                activation_embeds["layer 11"].append(t.mean(t.norm(output11, dim = -1), dim = 0))
                
        with open(f"data/gpt2_activation_embeds_{self.name}.pkl", "wb") as f:
            pickle.dump(activation_embeds, f)
                
        return activation_embeds

        
    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        norm_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": [],
            "layer 5": [],
            "layer 6": [],
            "layer 7": [],
            "layer 8": [],
            "layer 9": [],
            "layer 10": [],
            "layer 11": []
        }
        
        
        norm_actemb["layer 0"] = np.mean(self.activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(self.activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(self.activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(self.activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(self.activation_embeds["layer 4"], axis=0)
        norm_actemb["layer 5"] = np.mean(self.activation_embeds["layer 5"], axis=0)
        norm_actemb["layer 6"] = np.mean(self.activation_embeds["layer 6"], axis=0)
        norm_actemb["layer 7"] = np.mean(self.activation_embeds["layer 7"], axis=0)
        norm_actemb["layer 8"] = np.mean(self.activation_embeds["layer 8"], axis=0)
        norm_actemb["layer 9"] = np.mean(self.activation_embeds["layer 9"], axis=0)
        norm_actemb["layer 10"] = np.mean(self.activation_embeds["layer 10"], axis=0)
        norm_actemb["layer 11"] = np.mean(self.activation_embeds["layer 11"], axis=0)
        
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        actlist = np.array([
            np.log(np.array(norm_actemb["layer 0"])),
            np.log(np.array(norm_actemb["layer 1"])),
            np.log(np.array(norm_actemb["layer 2"])),
            np.log(np.array(norm_actemb["layer 3"])),
            np.log(np.array(norm_actemb["layer 4"])),
            np.log(np.array(norm_actemb["layer 5"])),
            np.log(np.array(norm_actemb["layer 6"])),
            np.log(np.array(norm_actemb["layer 7"])),
            np.log(np.array(norm_actemb["layer 8"])),
            np.log(np.array(norm_actemb["layer 9"])),
            np.log(np.array(norm_actemb["layer 10"])),
            np.log(np.array(norm_actemb["layer 11"])),
            # mean_acts["last layer"]
            ])
        
        print(actlist)
        self.plotting(data=actlist, name = f"figures/gpt2_activation_embeds_{self.name}.png")


    def plotting(self, data, name):
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(10, 11))  # Set figure size
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
        plt.title(f"[GPT-2]:Activation of {self.title}")

        # Show the heatmap
        plt.savefig(name)
        plt.close()




def img_concat():

    # List of image file paths (assuming you have 22 image paths)
    image_paths = ["mfigures_norm/" + img for img in os.listdir("mfigures_norm")]

    # Determine the grid size, for example, 5 rows and 5 columns
    rows = 5
    cols = 4

    # Create a figure to hold the grid
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))

    # Flatten the axes array for easy iteration
    axes = axes.flatten()

    # Loop over images and axes and plot each image
    for i, (img_path, ax) in enumerate(zip(image_paths, axes)):
        if i < len(image_paths):
            img = mpimg.imread(img_path)
            ax.imshow(img)
        ax.axis('off')  # Hide the axes

    # Save the combined image or display it
    plt.tight_layout()
    plt.savefig("mfigures_norm/combined_image.png")


class single_sample_act_norm:
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
    
    def activation(self):
        
        act_dict = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": [],
            "layer 5": [],
            "layer 6": [],
            "layer 7": [],
            "layer 8": [],
            "layer 9": [],
            "layer 10": []
        }
        random_indices = random.sample(range(len(self.data)), 20)
        random_samples = [self.data[i] for i in random_indices]
        for index, sample in enumerate(random_samples):
            with self.model.trace(sample['text']) as tracer:
                output0 = self.model.transformer.h[0].output[0].save()
                output1 = self.model.transformer.h[1].output[0].save()
                output2 = self.model.transformer.h[2].output[0].save()
                output3 = self.model.transformer.h[3].output[0].save()
                output4 = self.model.transformer.h[4].output[0].save()
                output5 = self.model.transformer.h[5].output[0].save()
                output6 = self.model.transformer.h[6].output[0].save()
                output7 = self.model.transformer.h[7].output[0].save()
                output8 = self.model.transformer.h[8].output[0].save()
                output9 = self.model.transformer.h[9].output[0].save()
                output10 = self.model.transformer.h[10].output[0].save()
                # output = self.model.embed_out.output.save()
            
            for i,j in enumerate([output0, output1, output2, output3, output4, output5, output6, output7, output8, output9, output10]):    
                act_dict[f"layer {i}"].append(np.array([t.norm(j, dim = -1).detach()]))
            
        return act_dict
    
    def norm(self):
        
        activations = self.activation()
        for index in range(len(activations["layer 0"])):
            data = np.array(
                [np.log(activations["layer 0"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 1"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 2"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 3"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 4"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 5"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 6"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 7"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 8"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 9"][index].squeeze(0).squeeze(0)),
                np.log(activations["layer 10"][index].squeeze(0).squeeze(0)),
                ]
            )
                
            self.plot(data=data, name=f"mfigures_norm/single_sample_{index}.png")
    
    def plot(self, data, name):
        plt.figure(figsize=(10, 11))
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        plt.title('Single Sample Token activations in different layers')
        plt.savefig(name)
        plt.close()
            

class single_sample_grad_norm:
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.model.train()
    def gradients(self):
        
        act_dict = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
            
        random_indices = random.sample(range(len(self.data)), 20)
        random_samples = [self.data[i] for i in random_indices]
        for index, sample in enumerate(random_samples):
            pprint(sample['text'])
            print()
            with self.model.trace(sample['text']) as tracer:
                output0 = self.model.gpt_neox.layers[0].output[0].grad.save()
                output1 = self.model.gpt_neox.layers[1].output[0].grad.save()
                output2 = self.model.gpt_neox.layers[2].output[0].grad.save()
                output3 = self.model.gpt_neox.layers[3].output[0].grad.save()
                output4 = self.model.gpt_neox.layers[4].output[0].grad.save()
                self.model.output.logits.sum().backward()
            
            for i,j in enumerate([output0, output1, output2, output3, output4]):
                act_dict[f"layer {i}"].append(np.array([t.norm(j, dim = -1).detach()]))
        print(output0)
        print(output1)
        print(output2)
        print(output3)
        print(output4)
        return act_dict
    
    def grad_norm(self):
        
        gradients = self.gradients()
        for index in range(len(gradients["layer 0"])):
            grad_data = np.array(
                [gradients["layer 0"][index].squeeze(0).squeeze(0),
                gradients["layer 1"][index].squeeze(0).squeeze(0),
                gradients["layer 2"][index].squeeze(0).squeeze(0),
                gradients["layer 3"][index].squeeze(0).squeeze(0),
                gradients["layer 4"][index].squeeze(0).squeeze(0)]
            )
                
            self.plot(data=grad_data, name=f"mfigures_gradnorm/single_sample_{index}.png")
    
    def plot(self, data, name):
        plt.figure(figsize=(10, 5))
        plt.imshow(data, aspect='auto', cmap='viridis')
        plt.colorbar()
        plt.xlabel('Tokens')
        plt.ylabel('Layers')
        plt.title('Single Sample Token activations in different layers')
        plt.savefig(name)
        plt.close()


def plotting(data, name):
    # Create the heatmap
    plt.figure(figsize=(10, 5))  # Set figure size
    plt.imshow(data, aspect='auto', cmap='viridis')  # Choose a color map like 'viridis', 'plasma', etc.

    # Add color bar to indicate the scale
    plt.colorbar()

    # Set labels
    plt.xlabel('Tokens')
    plt.ylabel('Layers')

    # Optionally, you can add titles
    if name == "figures/layer_seq_norm.png":
        plt.title('Token activations in different layers')
    elif name == "figures/layer_seq_normwmean.png":
        plt.title('Diff Mean Token activations in different layers')
    elif name == "figures/grads_layer_seq_norm.png":
        plt.title('Token gradients in different layers')
    elif name == "figures/grad_layer_seq_normwmean.png":
        plt.title('Diff Mean Token gradients in different layers')

    # Show the heatmap
    plt.savefig(name)
    plt.close()
