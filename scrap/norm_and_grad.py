from scrap.imports import *


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

    def act_normwmean_fn_ppma(self):
        
        normwmean_actemb_ppma = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
                output0_ppma = self.model.gpt_neox.layers[0].output[0].save()
                output1_ppma = self.model.gpt_neox.layers[1].output[0].save()
                output2_ppma = self.model.gpt_neox.layers[2].output[0].save()
                output3_ppma = self.model.gpt_neox.layers[3].output[0].save()
                output4_ppma = self.model.gpt_neox.layers[4].output[0].save()
            
            output0_ppma = output0_ppma.detach()
            output1_ppma = output1_ppma.detach()
            output2_ppma = output2_ppma.detach()
            output3_ppma = output3_ppma.detach()
            output4_ppma = output4_ppma.detach()

            output0mean_ppma = output0_ppma - t.mean(output0_ppma, dim = 0, keepdim = True)
            output1mean_ppma = output1_ppma - t.mean(output1_ppma, dim = 0, keepdim = True)
            output2mean_ppma = output2_ppma - t.mean(output2_ppma, dim = 0, keepdim = True)
            output3mean_ppma = output3_ppma - t.mean(output3_ppma, dim = 0, keepdim = True)
            output4mean_ppma = output4_ppma - t.mean(output4_ppma, dim = 0, keepdim = True)
            
            normwmean_actemb_ppma["layer 0"].append(t.mean(t.norm(output0mean_ppma, dim = -1), dim = 0))
            normwmean_actemb_ppma["layer 1"].append(t.mean(t.norm(output1mean_ppma, dim = -1), dim = 0))
            normwmean_actemb_ppma["layer 2"].append(t.mean(t.norm(output2mean_ppma, dim = -1), dim = 0))
            normwmean_actemb_ppma["layer 3"].append(t.mean(t.norm(output3mean_ppma, dim = -1), dim = 0))
            normwmean_actemb_ppma["layer 4"].append(t.mean(t.norm(output4mean_ppma, dim = -1), dim = 0))
            
            del output0_ppma, output1_ppma, output2_ppma, output3_ppma, output4_ppma
            gc.collect()
            
        return normwmean_actemb_ppma

    def act_normwmean_ppma(self):
        
        try:
            with open(f"data/pythia_wmean_activation_embeds_{self.name}.pkl", "rb") as f:
                wmean_actemb_ppma = pickle.load(f)
        except:
            with open(f"data/pythia_wmean_activation_embeds_{self.name}.pkl", "wb") as f:
                wmean_actemb_ppma = self.act_normwmean_fn_ppma()
                pickle.dump(wmean_actemb_ppma, f)
        
        normwmean_actemb_ppma = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        print(np.array(wmean_actemb_ppma["layer 0"]).shape)
        normwmean_actemb_ppma["layer 0"] = np.mean(np.array(wmean_actemb_ppma["layer 0"]), axis=0)
        normwmean_actemb_ppma["layer 1"] = np.mean(np.array(wmean_actemb_ppma["layer 1"]), axis=0)
        normwmean_actemb_ppma["layer 2"] = np.mean(np.array(wmean_actemb_ppma["layer 2"]), axis=0)
        normwmean_actemb_ppma["layer 3"] = np.mean(np.array(wmean_actemb_ppma["layer 3"]), axis=0)
        normwmean_actemb_ppma["layer 4"] = np.mean(np.array(wmean_actemb_ppma["layer 4"]), axis=0)

        actlistmean_ppma = np.array([
            np.log(np.array(normwmean_actemb_ppma["layer 0"])),
            np.log(np.array(normwmean_actemb_ppma["layer 1"])),
            np.log(np.array(normwmean_actemb_ppma["layer 2"])),
            np.log(np.array(normwmean_actemb_ppma["layer 3"])),
            np.log(np.array(normwmean_actemb_ppma["layer 4"])),
            ])

        self.plotting(data=actlistmean_ppma, name = f"figures/pythia_wmean_activation_embeds_{self.name}.png")

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
        self.plotting_grad(data=gradlist, name = f"figures/pythia_grad_embed_{self.name}.png")
    
    def gradwmean_fn_ppma(self):
        
        normwmean_grademb_ppma = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
                output0_grad_ppma = self.model.gpt_neox.layers[0].output[0].grad.save()
                output1_grad_ppma = self.model.gpt_neox.layers[1].output[0].grad.save()
                output2_grad_ppma = self.model.gpt_neox.layers[2].output[0].grad.save()
                output3_grad_ppma = self.model.gpt_neox.layers[3].output[0].grad.save()
                output4_grad_ppma = self.model.gpt_neox.layers[4].output[0].grad.save()
                
                self.model.output.logits.sum().backward()
            
            output0_grad_ppma = output0_grad_ppma.detach()
            output1_grad_ppma = output1_grad_ppma.detach()
            output2_grad_ppma = output2_grad_ppma.detach()
            output3_grad_ppma = output3_grad_ppma.detach()
            output4_grad_ppma = output4_grad_ppma.detach()
                
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            output0mean_ppma = output0_grad_ppma - t.mean(output0_grad_ppma, dim = 0, keepdim = True)
            output1mean_ppma = output1_grad_ppma - t.mean(output1_grad_ppma, dim = 0, keepdim = True)
            output2mean_ppma = output2_grad_ppma - t.mean(output2_grad_ppma, dim = 0, keepdim = True)
            output3mean_ppma = output3_grad_ppma - t.mean(output3_grad_ppma, dim = 0, keepdim = True)
            output4mean_ppma = output4_grad_ppma - t.mean(output4_grad_ppma, dim = 0, keepdim = True)
            
            normwmean_grademb_ppma["layer 0"].append(t.mean(t.norm(output0mean_ppma, dim = -1), dim = 0))
            normwmean_grademb_ppma["layer 1"].append(t.mean(t.norm(output1mean_ppma, dim = -1), dim = 0))
            normwmean_grademb_ppma["layer 2"].append(t.mean(t.norm(output2mean_ppma, dim = -1), dim = 0))
            normwmean_grademb_ppma["layer 3"].append(t.mean(t.norm(output3mean_ppma, dim = -1), dim = 0))
            normwmean_grademb_ppma["layer 4"].append(t.mean(t.norm(output4mean_ppma, dim = -1), dim = 0))
            
        return normwmean_grademb_ppma

    def grad_normwmean_ppma(self):
        
        try:
            with open(f"data/pythia_wmean_grad_embeds_{self.name}.pkl", "rb") as f:
                wmean_grademb_ppma = pickle.load(f)
        except:
            with open(f"data/pythia_wmean_grad_embeds_{self.name}.pkl", "wb") as f:
                wmean_grademb_ppma = self.gradwmean_fn_ppma()
                pickle.dump(wmean_grademb_ppma, f)
        
        normwmean_grademb_ppma = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        normwmean_grademb_ppma["layer 0"] = np.mean(np.array(wmean_grademb_ppma["layer 0"]), axis=0)
        normwmean_grademb_ppma["layer 1"] = np.mean(np.array(wmean_grademb_ppma["layer 1"]), axis=0)
        normwmean_grademb_ppma["layer 2"] = np.mean(np.array(wmean_grademb_ppma["layer 2"]), axis=0)
        normwmean_grademb_ppma["layer 3"] = np.mean(np.array(wmean_grademb_ppma["layer 3"]), axis=0)
        normwmean_grademb_ppma["layer 4"] = np.mean(np.array(wmean_grademb_ppma["layer 4"]), axis=0)

        gradlistmean_grad_ppma = np.array([
            np.log(np.array(normwmean_grademb_ppma["layer 0"])),
            np.log(np.array(normwmean_grademb_ppma["layer 1"])),
            np.log(np.array(normwmean_grademb_ppma["layer 2"])),
            np.log(np.array(normwmean_grademb_ppma["layer 3"])),
            np.log(np.array(normwmean_grademb_ppma["layer 4"])),
            ])

        self.plotting_grad(
            data=gradlistmean_grad_ppma, 
            name = f"figures/pythia_wmean_gradients_embeds_{self.name}.png"
            )

    def plotting_grad(self,data, name):
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

    def act_normwmean_fn_mlp(self):
        
        normwmean_actemb_mlp = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
                output0_mlp = self.model.gpt_neox.layers[0].mlp.output.save()
                output1_mlp = self.model.gpt_neox.layers[1].mlp.output.save()
                output2_mlp = self.model.gpt_neox.layers[2].mlp.output.save()
                output3_mlp = self.model.gpt_neox.layers[3].mlp.output.save()
                output4_mlp = self.model.gpt_neox.layers[4].mlp.output.save()
            
            output0_mlp  = output0_mlp.detach()
            output1_mlp  = output1_mlp.detach()
            output2_mlp  = output2_mlp.detach()
            output3_mlp  = output3_mlp.detach()
            output4_mlp  = output4_mlp.detach()

            output0mean_mlp  = output0_mlp - t.mean(output0_mlp, dim = 0, keepdim = True)
            output1mean_mlp  = output1_mlp - t.mean(output1_mlp, dim = 0, keepdim = True)
            output2mean_mlp  = output2_mlp - t.mean(output2_mlp, dim = 0, keepdim = True)
            output3mean_mlp  = output3_mlp - t.mean(output3_mlp, dim = 0, keepdim = True)
            output4mean_mlp  = output4_mlp - t.mean(output4_mlp, dim = 0, keepdim = True)
            
            normwmean_actemb_mlp["layer 0"].append(t.mean(t.norm(output0mean_mlp, dim = -1), dim = 0))
            normwmean_actemb_mlp["layer 1"].append(t.mean(t.norm(output1mean_mlp, dim = -1), dim = 0))
            normwmean_actemb_mlp["layer 2"].append(t.mean(t.norm(output2mean_mlp, dim = -1), dim = 0))
            normwmean_actemb_mlp["layer 3"].append(t.mean(t.norm(output3mean_mlp, dim = -1), dim = 0))
            normwmean_actemb_mlp["layer 4"].append(t.mean(t.norm(output4mean_mlp, dim = -1), dim = 0))
            
            del output0_mlp, output1_mlp, output2_mlp, output3_mlp, output4_mlp
            gc.collect()
            
        return normwmean_actemb_mlp

    def act_normwmean_mlp(self):
        
        try:
            with open(f"data/pythia_wmean_activation_embeds_{self.name}.pkl", "rb") as f:
                wmean_actemb_mlp = pickle.load(f)
        except:
            with open(f"data/pythia_wmean_activation_embeds_{self.name}.pkl", "wb") as f:
                wmean_actemb_mlp = self.act_normwmean_fn_mlp()
                pickle.dump(wmean_actemb_mlp, f)
        
        normwmean_actemb_mlp = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        normwmean_actemb_mlp["layer 0"] = np.mean(np.array(wmean_actemb_mlp["layer 0"]), axis=0)
        normwmean_actemb_mlp["layer 1"] = np.mean(np.array(wmean_actemb_mlp["layer 1"]), axis=0)
        normwmean_actemb_mlp["layer 2"] = np.mean(np.array(wmean_actemb_mlp["layer 2"]), axis=0)
        normwmean_actemb_mlp["layer 3"] = np.mean(np.array(wmean_actemb_mlp["layer 3"]), axis=0)
        normwmean_actemb_mlp["layer 4"] = np.mean(np.array(wmean_actemb_mlp["layer 4"]), axis=0)

        actlistmean_mlp = np.array([
            np.log(np.array(normwmean_actemb_mlp["layer 0"])),
            np.log(np.array(normwmean_actemb_mlp["layer 1"])),
            np.log(np.array(normwmean_actemb_mlp["layer 2"])),
            np.log(np.array(normwmean_actemb_mlp["layer 3"])),
            np.log(np.array(normwmean_actemb_mlp["layer 4"])),
            ])

        self.plotting(data=actlistmean_mlp, name = f"figures/pythia_wmean_activation_embeds_{self.name}.png")


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
        self.plotting_grad(data=gradlist, name = f"figures/pythia_grad_embed_{self.name}.png")

    def gradwmean_fn_mlp(self):
        
        normwmean_grademb_mlp = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
                output0_grad_mlp = self.model.gpt_neox.layers[0].mlp.output.grad.save()
                output1_grad_mlp = self.model.gpt_neox.layers[1].mlp.output.grad.save()
                output2_grad_mlp = self.model.gpt_neox.layers[2].mlp.output.grad.save()
                output3_grad_mlp = self.model.gpt_neox.layers[3].mlp.output.grad.save()
                output4_grad_mlp = self.model.gpt_neox.layers[4].mlp.output.grad.save()
                
                self.model.output.logits.sum().backward()
            
            output0_grad_mlp = output0_grad_mlp.detach()
            output1_grad_mlp = output1_grad_mlp.detach()
            output2_grad_mlp = output2_grad_mlp.detach()
            output3_grad_mlp = output3_grad_mlp.detach()
            output4_grad_mlp = output4_grad_mlp.detach()
                
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            output0mean_mlp = output0_grad_mlp - t.mean(output0_grad_mlp, dim = 0, keepdim = True)
            output1mean_mlp = output1_grad_mlp - t.mean(output1_grad_mlp, dim = 0, keepdim = True)
            output2mean_mlp = output2_grad_mlp - t.mean(output2_grad_mlp, dim = 0, keepdim = True)
            output3mean_mlp = output3_grad_mlp - t.mean(output3_grad_mlp, dim = 0, keepdim = True)
            output4mean_mlp = output4_grad_mlp - t.mean(output4_grad_mlp, dim = 0, keepdim = True)
            
            normwmean_grademb_mlp["layer 0"].append(t.mean(t.norm(output0mean_mlp, dim = -1), dim = 0))
            normwmean_grademb_mlp["layer 1"].append(t.mean(t.norm(output1mean_mlp, dim = -1), dim = 0))
            normwmean_grademb_mlp["layer 2"].append(t.mean(t.norm(output2mean_mlp, dim = -1), dim = 0))
            normwmean_grademb_mlp["layer 3"].append(t.mean(t.norm(output3mean_mlp, dim = -1), dim = 0))
            normwmean_grademb_mlp["layer 4"].append(t.mean(t.norm(output4mean_mlp, dim = -1), dim = 0))
            
            
        return normwmean_grademb_mlp

    def grad_normwmean_mlp(self):
        
        try:
            with open(f"data/pythia_wmean_grad_embeds_{self.name}.pkl", "rb") as f:
                wmean_grademb_mlp = pickle.load(f)
        except:
            with open(f"data/pythia_wmean_grad_embeds_{self.name}.pkl", "wb") as f:
                wmean_grademb_mlp = self.gradwmean_fn_mlp()
                pickle.dump(wmean_grademb_mlp, f)
        
        normwmean_grademb_mlp = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        normwmean_grademb_mlp["layer 0"] = np.mean(np.array(wmean_grademb_mlp["layer 0"]), axis=0)
        normwmean_grademb_mlp["layer 1"] = np.mean(np.array(wmean_grademb_mlp["layer 1"]), axis=0)
        normwmean_grademb_mlp["layer 2"] = np.mean(np.array(wmean_grademb_mlp["layer 2"]), axis=0)
        normwmean_grademb_mlp["layer 3"] = np.mean(np.array(wmean_grademb_mlp["layer 3"]), axis=0)
        normwmean_grademb_mlp["layer 4"] = np.mean(np.array(wmean_grademb_mlp["layer 4"]), axis=0)

        gradlistmean_grad_mlp = np.array([
            np.log(np.array(normwmean_grademb_mlp["layer 0"])),
            np.log(np.array(normwmean_grademb_mlp["layer 1"])),
            np.log(np.array(normwmean_grademb_mlp["layer 2"])),
            np.log(np.array(normwmean_grademb_mlp["layer 3"])),
            np.log(np.array(normwmean_grademb_mlp["layer 4"])),
            ])

        self.plotting_grad(
            data=gradlistmean_grad_mlp, 
            name = f"figures/pythia_wmean_gradients_embeds_{self.name}.png"
            )


    def plotting_grad(self, data, name):
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

    def act_normwmean_fn_attn(self):
        
        normwmean_actemb_attn = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
                output0_attn = self.model.gpt_neox.layers[0].attention.output[0].save()
                output1_attn = self.model.gpt_neox.layers[1].attention.output[0].save()
                output2_attn = self.model.gpt_neox.layers[2].attention.output[0].save()
                output3_attn = self.model.gpt_neox.layers[3].attention.output[0].save()
                output4_attn = self.model.gpt_neox.layers[4].attention.output[0].save()
                
            
            output0_attn  = output0_attn.detach()
            output1_attn  = output1_attn.detach()
            output2_attn  = output2_attn.detach()
            output3_attn  = output3_attn.detach()
            output4_attn  = output4_attn.detach()

            output0mean_attn  = output0_attn - t.mean(output0_attn, dim = 0, keepdim = True)
            output1mean_attn  = output1_attn - t.mean(output1_attn, dim = 0, keepdim = True)
            output2mean_attn  = output2_attn - t.mean(output2_attn, dim = 0, keepdim = True)
            output3mean_attn  = output3_attn - t.mean(output3_attn, dim = 0, keepdim = True)
            output4mean_attn  = output4_attn - t.mean(output4_attn, dim = 0, keepdim = True)
            
            normwmean_actemb_attn["layer 0"].append(t.mean(t.norm(output0mean_attn, dim = -1), dim = 0))
            normwmean_actemb_attn["layer 1"].append(t.mean(t.norm(output1mean_attn, dim = -1), dim = 0))
            normwmean_actemb_attn["layer 2"].append(t.mean(t.norm(output2mean_attn, dim = -1), dim = 0))
            normwmean_actemb_attn["layer 3"].append(t.mean(t.norm(output3mean_attn, dim = -1), dim = 0))
            normwmean_actemb_attn["layer 4"].append(t.mean(t.norm(output4mean_attn, dim = -1), dim = 0))
            
            del output0_attn, output1_attn, output2_attn, output3_attn, output4_attn
            gc.collect()
            
        return normwmean_actemb_attn

    def act_normwmean_attn(self):
        
        try:
            with open(f"data/pythia_wmean_activation_embeds_{self.name}.pkl", "rb") as f:
                wmean_actemb_attn = pickle.load(f)
        except:
            with open(f"data/pythia_wmean_activation_embeds_{self.name}.pkl", "wb") as f:
                wmean_actemb_attn = self.act_normwmean_fn_attn()
                pickle.dump(wmean_actemb_attn, f)
        
        normwmean_actemb_attn = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        print(np.array(wmean_actemb_attn["layer 0"]).shape)
        normwmean_actemb_attn["layer 0"] = np.mean(np.array(wmean_actemb_attn["layer 0"]), axis=0)
        normwmean_actemb_attn["layer 1"] = np.mean(np.array(wmean_actemb_attn["layer 1"]), axis=0)
        normwmean_actemb_attn["layer 2"] = np.mean(np.array(wmean_actemb_attn["layer 2"]), axis=0)
        normwmean_actemb_attn["layer 3"] = np.mean(np.array(wmean_actemb_attn["layer 3"]), axis=0)
        normwmean_actemb_attn["layer 4"] = np.mean(np.array(wmean_actemb_attn["layer 4"]), axis=0)

        actlistmean_attn = np.array([
            np.log(np.array(normwmean_actemb_attn["layer 0"])),
            np.log(np.array(normwmean_actemb_attn["layer 1"])),
            np.log(np.array(normwmean_actemb_attn["layer 2"])),
            np.log(np.array(normwmean_actemb_attn["layer 3"])),
            np.log(np.array(normwmean_actemb_attn["layer 4"])),
            ])

        self.plotting(data=actlistmean_attn, name = f"figures/pythia_wmean_activation_embeds_{self.name}.png")


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
        self.plotting_grad(data=gradlist, name = f"figures/pythia_grad_embed_{self.name}.png")


    def gradwmean_fn_attn(self):
        
        normwmean_grademb_attn = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
                output0_grad_attn = self.model.gpt_neox.layers[0].attention.output[0].grad.save()
                output1_grad_attn = self.model.gpt_neox.layers[1].attention.output[0].grad.save()
                output2_grad_attn = self.model.gpt_neox.layers[2].attention.output[0].grad.save()
                output3_grad_attn = self.model.gpt_neox.layers[3].attention.output[0].grad.save()
                output4_grad_attn = self.model.gpt_neox.layers[4].attention.output[0].grad.save()
                
                self.model.output.logits.sum().backward()
            
            output0_grad_attn = output0_grad_attn.detach()
            output1_grad_attn = output1_grad_attn.detach()
            output2_grad_attn = output2_grad_attn.detach()
            output3_grad_attn = output3_grad_attn.detach()
            output4_grad_attn = output4_grad_attn.detach()
                
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            output0mean_attn = output0_grad_attn - t.mean(output0_grad_attn, dim = 0, keepdim = True)
            output1mean_attn = output1_grad_attn - t.mean(output1_grad_attn, dim = 0, keepdim = True)
            output2mean_attn = output2_grad_attn - t.mean(output2_grad_attn, dim = 0, keepdim = True)
            output3mean_attn = output3_grad_attn - t.mean(output3_grad_attn, dim = 0, keepdim = True)
            output4mean_attn = output4_grad_attn - t.mean(output4_grad_attn, dim = 0, keepdim = True)
            
            normwmean_grademb_attn["layer 0"].append(t.mean(t.norm(output0mean_attn, dim = -1), dim = 0))
            normwmean_grademb_attn["layer 1"].append(t.mean(t.norm(output1mean_attn, dim = -1), dim = 0))
            normwmean_grademb_attn["layer 2"].append(t.mean(t.norm(output2mean_attn, dim = -1), dim = 0))
            normwmean_grademb_attn["layer 3"].append(t.mean(t.norm(output3mean_attn, dim = -1), dim = 0))
            normwmean_grademb_attn["layer 4"].append(t.mean(t.norm(output4mean_attn, dim = -1), dim = 0))
            
            
        return normwmean_grademb_attn

    def grad_normwmean_attn(self):
        
        try:
            with open(f"data/pythia_wmean_grad_embeds_{self.name}.pkl", "rb") as f:
                wmean_grademb_attn = pickle.load(f)
        except:
            with open(f"data/pythia_wmean_grad_embeds_{self.name}.pkl", "wb") as f:
                wmean_grademb_attn = self.gradwmean_fn_attn()
                pickle.dump(wmean_grademb_attn, f)
        
        normwmean_grademb_attn = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        normwmean_grademb_attn["layer 0"] = np.mean(np.array(wmean_grademb_attn["layer 0"]), axis=0)
        normwmean_grademb_attn["layer 1"] = np.mean(np.array(wmean_grademb_attn["layer 1"]), axis=0)
        normwmean_grademb_attn["layer 2"] = np.mean(np.array(wmean_grademb_attn["layer 2"]), axis=0)
        normwmean_grademb_attn["layer 3"] = np.mean(np.array(wmean_grademb_attn["layer 3"]), axis=0)
        normwmean_grademb_attn["layer 4"] = np.mean(np.array(wmean_grademb_attn["layer 4"]), axis=0)

        gradlistmean_grad_attn = np.array([
            np.log(np.array(normwmean_grademb_attn["layer 0"])),
            np.log(np.array(normwmean_grademb_attn["layer 1"])),
            np.log(np.array(normwmean_grademb_attn["layer 2"])),
            np.log(np.array(normwmean_grademb_attn["layer 3"])),
            np.log(np.array(normwmean_grademb_attn["layer 4"])),
            ])

        self.plotting_grad(
            data=gradlistmean_grad_attn, 
            name = f"figures/pythia_wmean_gradients_embeds_{self.name}.png"
            )


    def plotting_grad(self, data, name):
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


class grad_gpt2_resid_post_mlp_addn:
    
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
            with open(f"data/gpt2_grad_norm_{name}.pkl", "rb") as f:
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
        "layer 4": [],
        "layer 5": [],
        "layer 6": [],
        "layer 7": [],
        "layer 8": [],
        "layer 9": [],
        "layer 10": [],
        "layer 11": []
        }
        
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
            
                output0 = self.model.transformer.h[0].output[0].grad.save()
                output1 = self.model.transformer.h[1].output[0].grad.save()
                output2 = self.model.transformer.h[2].output[0].grad.save()
                output3 = self.model.transformer.h[3].output[0].grad.save()
                output4 = self.model.transformer.h[4].output[0].grad.save()
                output5 = self.model.transformer.h[5].output[0].grad.save()
                output6 = self.model.transformer.h[6].output[0].grad.save()
                output7 = self.model.transformer.h[7].output[0].grad.save()
                output8 = self.model.transformer.h[8].output[0].grad.save()
                output9 = self.model.transformer.h[9].output[0].grad.save()
                output10 = self.model.transformer.h[10].output[0].grad.save()
                output11 = self.model.transformer.h[11].output[0].grad.save()
                
                self.model.output.logits.sum().backward()
            
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            grad_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
            grad_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
            grad_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
            grad_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
            grad_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
            grad_embeds["layer 5"].append(t.mean(t.norm(output5, dim = -1), dim = 0))
            grad_embeds["layer 6"].append(t.mean(t.norm(output6, dim = -1), dim = 0))
            grad_embeds["layer 7"].append(t.mean(t.norm(output7, dim = -1), dim = 0))
            grad_embeds["layer 8"].append(t.mean(t.norm(output8, dim = -1), dim = 0))
            grad_embeds["layer 9"].append(t.mean(t.norm(output9, dim = -1), dim = 0))
            grad_embeds["layer 10"].append(t.mean(t.norm(output10, dim = -1), dim = 0))
            grad_embeds["layer 11"].append(t.mean(t.norm(output11, dim = -1), dim = 0))
            
        with open(f"data/gpt2_grad_norm_{self.name}.pkl", "wb") as f:
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
            "layer 4": [],
            "layer 5": [],
            "layer 6": [],
            "layer 7": [],
            "layer 8": [],
            "layer 9": [],
            "layer 10": [],
            "layer 11": []
        }
        
        
        grad_actemb["layer 0"] = np.mean(self.grads["layer 0"], axis=0)
        grad_actemb["layer 1"] = np.mean(self.grads["layer 1"], axis=0)
        grad_actemb["layer 2"] = np.mean(self.grads["layer 2"], axis=0)
        grad_actemb["layer 3"] = np.mean(self.grads["layer 3"], axis=0)
        grad_actemb["layer 4"] = np.mean(self.grads["layer 4"], axis=0)
        grad_actemb["layer 5"] = np.mean(self.grads["layer 5"], axis=0)
        grad_actemb["layer 6"] = np.mean(self.grads["layer 6"], axis=0)
        grad_actemb["layer 7"] = np.mean(self.grads["layer 7"], axis=0)
        grad_actemb["layer 8"] = np.mean(self.grads["layer 8"], axis=0)
        grad_actemb["layer 9"] = np.mean(self.grads["layer 9"], axis=0)
        grad_actemb["layer 10"] = np.mean(self.grads["layer 10"], axis=0)
        grad_actemb["layer 11"] = np.mean(self.grads["layer 11"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        gradlist = np.array([
            np.log(np.array(grad_actemb["layer 0"])),
            np.log(np.array(grad_actemb["layer 1"])),
            np.log(np.array(grad_actemb["layer 2"])),
            np.log(np.array(grad_actemb["layer 3"])),
            np.log(np.array(grad_actemb["layer 4"])),
            np.log(np.array(grad_actemb["layer 5"])),
            np.log(np.array(grad_actemb["layer 6"])),
            np.log(np.array(grad_actemb["layer 7"])),
            np.log(np.array(grad_actemb["layer 8"])),
            np.log(np.array(grad_actemb["layer 9"])),
            np.log(np.array(grad_actemb["layer 10"])),
            np.log(np.array(grad_actemb["layer 11"])),
            # mean_acts["last layer"]
            ])
        print(gradlist)
        self.plotting(data=gradlist, name = f"figures/gpt2_grad_embed_{self.name}.png")
    

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
        plt.title(f"[GPT-2]:Gradient of {self.title}")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_gpt2_mlp:

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
                    output0 = self.model.transformer.h[0].mlp.output.save()
                    output1 = self.model.transformer.h[1].mlp.output.save()
                    output2 = self.model.transformer.h[2].mlp.output.save()
                    output3 = self.model.transformer.h[3].mlp.output.save()
                    output4 = self.model.transformer.h[4].mlp.output.save()
                    output5 = self.model.transformer.h[5].mlp.output.save()
                    output6 = self.model.transformer.h[6].mlp.output.save()
                    output7 = self.model.transformer.h[7].mlp.output.save()
                    output8 = self.model.transformer.h[8].mlp.output.save()
                    output9 = self.model.transformer.h[9].mlp.output.save()
                    output10 = self.model.transformer.h[10].mlp.output.save()
                    output11 = self.model.transformer.h[11].mlp.output.save()

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


class grad_gpt2_mlp:
    
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
            with open(f"data/gpt2_grad_norm_{name}.pkl", "rb") as f:
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
        "layer 4": [],
        "layer 5": [],
        "layer 6": [],
        "layer 7": [],
        "layer 8": [],
        "layer 9": [],
        "layer 10": [],
        "layer 11": []
        }
        
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
            
                output0 = self.model.transformer.h[0].mlp.output.grad.save()
                output1 = self.model.transformer.h[1].mlp.output.grad.save()
                output2 = self.model.transformer.h[2].mlp.output.grad.save()
                output3 = self.model.transformer.h[3].mlp.output.grad.save()
                output4 = self.model.transformer.h[4].mlp.output.grad.save()
                output5 = self.model.transformer.h[5].mlp.output.grad.save()
                output6 = self.model.transformer.h[6].mlp.output.grad.save()
                output7 = self.model.transformer.h[7].mlp.output.grad.save()
                output8 = self.model.transformer.h[8].mlp.output.grad.save()
                output9 = self.model.transformer.h[9].mlp.output.grad.save()
                output10 = self.model.transformer.h[10].mlp.output.grad.save()
                output11 = self.model.transformer.h[11].mlp.output.grad.save()
                
                
                self.model.output.logits.sum().backward()
            
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            grad_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
            grad_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
            grad_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
            grad_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
            grad_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
            grad_embeds["layer 5"].append(t.mean(t.norm(output5, dim = -1), dim = 0))
            grad_embeds["layer 6"].append(t.mean(t.norm(output6, dim = -1), dim = 0))
            grad_embeds["layer 7"].append(t.mean(t.norm(output7, dim = -1), dim = 0))
            grad_embeds["layer 8"].append(t.mean(t.norm(output8, dim = -1), dim = 0))
            grad_embeds["layer 9"].append(t.mean(t.norm(output9, dim = -1), dim = 0))
            grad_embeds["layer 10"].append(t.mean(t.norm(output10, dim = -1), dim = 0))
            grad_embeds["layer 11"].append(t.mean(t.norm(output11, dim = -1), dim = 0))
            
        with open(f"data/gpt2_grad_norm_{self.name}.pkl", "wb") as f:
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
            "layer 4": [],
            "layer 5": [],
            "layer 6": [],
            "layer 7": [],
            "layer 8": [],
            "layer 9": [],
            "layer 10": [],
            "layer 11": []
        }
        
        
        grad_actemb["layer 0"] = np.mean(self.grads["layer 0"], axis=0)
        grad_actemb["layer 1"] = np.mean(self.grads["layer 1"], axis=0)
        grad_actemb["layer 2"] = np.mean(self.grads["layer 2"], axis=0)
        grad_actemb["layer 3"] = np.mean(self.grads["layer 3"], axis=0)
        grad_actemb["layer 4"] = np.mean(self.grads["layer 4"], axis=0)
        grad_actemb["layer 5"] = np.mean(self.grads["layer 5"], axis=0)
        grad_actemb["layer 6"] = np.mean(self.grads["layer 6"], axis=0)
        grad_actemb["layer 7"] = np.mean(self.grads["layer 7"], axis=0)
        grad_actemb["layer 8"] = np.mean(self.grads["layer 8"], axis=0)
        grad_actemb["layer 9"] = np.mean(self.grads["layer 9"], axis=0)
        grad_actemb["layer 10"] = np.mean(self.grads["layer 10"], axis=0)
        grad_actemb["layer 11"] = np.mean(self.grads["layer 11"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        gradlist = np.array([
            np.log(np.array(grad_actemb["layer 0"])),
            np.log(np.array(grad_actemb["layer 1"])),
            np.log(np.array(grad_actemb["layer 2"])),
            np.log(np.array(grad_actemb["layer 3"])),
            np.log(np.array(grad_actemb["layer 4"])),
            np.log(np.array(grad_actemb["layer 5"])),
            np.log(np.array(grad_actemb["layer 6"])),
            np.log(np.array(grad_actemb["layer 7"])),
            np.log(np.array(grad_actemb["layer 8"])),
            np.log(np.array(grad_actemb["layer 9"])),
            np.log(np.array(grad_actemb["layer 10"])),
            np.log(np.array(grad_actemb["layer 11"])),
            # mean_acts["last layer"]
            ])
        print(gradlist)
        self.plotting(data=gradlist, name = f"figures/gpt2_grad_embed_{self.name}.png")
    

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
        plt.title(f"[GPT-2]:Gradient of {self.title}")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_gpt2_attention:

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
                    output0 = self.model.transformer.h[0].attn.output[0].save()
                    output1 = self.model.transformer.h[1].attn.output[0].save()
                    output2 = self.model.transformer.h[2].attn.output[0].save()
                    output3 = self.model.transformer.h[3].attn.output[0].save()
                    output4 = self.model.transformer.h[4].attn.output[0].save()
                    output5 = self.model.transformer.h[5].attn.output[0].save()
                    output6 = self.model.transformer.h[6].attn.output[0].save()
                    output7 = self.model.transformer.h[7].attn.output[0].save()
                    output8 = self.model.transformer.h[8].attn.output[0].save()
                    output9 = self.model.transformer.h[9].attn.output[0].save()
                    output10 = self.model.transformer.h[10].attn.output[0].save()
                    output11 = self.model.transformer.h[11].attn.output[0].save()

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


class grad_gpt2_attention:
    
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
            with open(f"data/gpt2_grad_norm_{name}.pkl", "rb") as f:
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
        "layer 4": [],
        "layer 5": [],
        "layer 6": [],
        "layer 7": [],
        "layer 8": [],
        "layer 9": [],
        "layer 10": [],
        "layer 11": []
        }
        
        
        for batch in tqdm(self.dataloader):
            
            with self.model.trace(batch["input_ids"]) as tracer:
            
                output0 = self.model.transformer.h[0].attn.output[0].grad.save()
                output1 = self.model.transformer.h[1].attn.output[0].grad.save()
                output2 = self.model.transformer.h[2].attn.output[0].grad.save()
                output3 = self.model.transformer.h[3].attn.output[0].grad.save()
                output4 = self.model.transformer.h[4].attn.output[0].grad.save()
                output5 = self.model.transformer.h[5].attn.output[0].grad.save()
                output6 = self.model.transformer.h[6].attn.output[0].grad.save()
                output7 = self.model.transformer.h[7].attn.output[0].grad.save()
                output8 = self.model.transformer.h[8].attn.output[0].grad.save()
                output9 = self.model.transformer.h[9].attn.output[0].grad.save()
                output10 = self.model.transformer.h[10].attn.output[0].grad.save()
                output11 = self.model.transformer.h[11].attn.output[0].grad.save()
                
                
                self.model.output.logits.sum().backward()
            
            # firstly taking the norm for the batch of 2 and then for the dimension of every token
            grad_embeds["layer 0"].append(t.mean(t.norm(output0, dim = -1), dim = 0))
            grad_embeds["layer 1"].append(t.mean(t.norm(output1, dim = -1), dim = 0))
            grad_embeds["layer 2"].append(t.mean(t.norm(output2, dim = -1), dim = 0))
            grad_embeds["layer 3"].append(t.mean(t.norm(output3, dim = -1), dim = 0))
            grad_embeds["layer 4"].append(t.mean(t.norm(output4, dim = -1), dim = 0))
            grad_embeds["layer 5"].append(t.mean(t.norm(output5, dim = -1), dim = 0))
            grad_embeds["layer 6"].append(t.mean(t.norm(output6, dim = -1), dim = 0))
            grad_embeds["layer 7"].append(t.mean(t.norm(output7, dim = -1), dim = 0))
            grad_embeds["layer 8"].append(t.mean(t.norm(output8, dim = -1), dim = 0))
            grad_embeds["layer 9"].append(t.mean(t.norm(output9, dim = -1), dim = 0))
            grad_embeds["layer 10"].append(t.mean(t.norm(output10, dim = -1), dim = 0))
            grad_embeds["layer 11"].append(t.mean(t.norm(output11, dim = -1), dim = 0))
            
        with open(f"data/gpt2_grad_norm_{self.name}.pkl", "wb") as f:
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
            "layer 4": [],
            "layer 5": [],
            "layer 6": [],
            "layer 7": [],
            "layer 8": [],
            "layer 9": [],
            "layer 10": [],
            "layer 11": []
        }
        
        
        grad_actemb["layer 0"] = np.mean(self.grads["layer 0"], axis=0)
        grad_actemb["layer 1"] = np.mean(self.grads["layer 1"], axis=0)
        grad_actemb["layer 2"] = np.mean(self.grads["layer 2"], axis=0)
        grad_actemb["layer 3"] = np.mean(self.grads["layer 3"], axis=0)
        grad_actemb["layer 4"] = np.mean(self.grads["layer 4"], axis=0)
        grad_actemb["layer 5"] = np.mean(self.grads["layer 5"], axis=0)
        grad_actemb["layer 6"] = np.mean(self.grads["layer 6"], axis=0)
        grad_actemb["layer 7"] = np.mean(self.grads["layer 7"], axis=0)
        grad_actemb["layer 8"] = np.mean(self.grads["layer 8"], axis=0)
        grad_actemb["layer 9"] = np.mean(self.grads["layer 9"], axis=0)
        grad_actemb["layer 10"] = np.mean(self.grads["layer 10"], axis=0)
        grad_actemb["layer 11"] = np.mean(self.grads["layer 11"], axis=0)
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        gradlist = np.array([
            np.log(np.array(grad_actemb["layer 0"])),
            np.log(np.array(grad_actemb["layer 1"])),
            np.log(np.array(grad_actemb["layer 2"])),
            np.log(np.array(grad_actemb["layer 3"])),
            np.log(np.array(grad_actemb["layer 4"])),
            np.log(np.array(grad_actemb["layer 5"])),
            np.log(np.array(grad_actemb["layer 6"])),
            np.log(np.array(grad_actemb["layer 7"])),
            np.log(np.array(grad_actemb["layer 8"])),
            np.log(np.array(grad_actemb["layer 9"])),
            np.log(np.array(grad_actemb["layer 10"])),
            np.log(np.array(grad_actemb["layer 11"])),
            # mean_acts["last layer"]
            ])
        print(gradlist)
        self.plotting(data=gradlist, name = f"figures/gpt2_grad_embed_{self.name}.png")
    

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
        plt.title(f"[GPT-2]:Gradient of {self.title}")

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