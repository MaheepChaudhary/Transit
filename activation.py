from imports import *

class act_pythia14_70_resid_post_mlp_addn:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        print(model)
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
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
                
                # output0.shape -> (batch_size, 128, 2048)
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
        
        norm_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        
        norm_actemb["layer 0"] = np.mean(np.array(activation_embeds["layer 0"]), axis=0)
        norm_actemb["layer 1"] = np.mean(np.array(activation_embeds["layer 1"]), axis=0)
        norm_actemb["layer 2"] = np.mean(np.array(activation_embeds["layer 2"]), axis=0)
        norm_actemb["layer 3"] = np.mean(np.array(activation_embeds["layer 3"]), axis=0)
        norm_actemb["layer 4"] = np.mean(np.array(activation_embeds["layer 4"]), axis=0)
        norm_actemb["layer 5"] = np.mean(np.array(activation_embeds["layer 5"]), axis=0)
        
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        
        actlist = np.array([
            np.log(np.array(norm_actemb["layer 0"])),
            np.log(np.array(norm_actemb["layer 1"])),
            np.log(np.array(norm_actemb["layer 2"])),
            np.log(np.array(norm_actemb["layer 3"])),
            np.log(np.array(norm_actemb["layer 4"])),
            np.log(np.array(norm_actemb["layer 5"]))
            # mean_acts["last layer"]
            ])
        
        
        try:
            os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
            os.mkdir(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_resid.pkl", "wb") as f:
            pickle.dump(actlist, f)
        
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_resid.png")

    def act_normwmean_fn_ppma(self):
        
        normwmean_actemb_ppma = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": [],
            "layer 5": []
        }
        
        for sample in tqdm(self.data):
            
            inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
            
            with self.model.trace(inputs) as tracer:
                output0_ppma = self.model.gpt_neox.layers[0].output[0].save()
                output1_ppma = self.model.gpt_neox.layers[1].output[0].save()
                output2_ppma = self.model.gpt_neox.layers[2].output[0].save()
                output3_ppma = self.model.gpt_neox.layers[3].output[0].save()
                output4_ppma = self.model.gpt_neox.layers[4].output[0].save()
                output5_ppma = self.model.gpt_neox.layers[5].output[0].save()
            
            output0_ppma = output0_ppma.detach().cpu()
            output1_ppma = output1_ppma.detach().cpu()
            output2_ppma = output2_ppma.detach().cpu()
            output3_ppma = output3_ppma.detach().cpu()
            output4_ppma = output4_ppma.detach().cpu()
            output5_ppma = output5_ppma.detach().cpu()

            output0mean_ppma = output0_ppma - t.mean(output0_ppma, dim = 0, keepdim = True)
            output1mean_ppma = output1_ppma - t.mean(output1_ppma, dim = 0, keepdim = True)
            output2mean_ppma = output2_ppma - t.mean(output2_ppma, dim = 0, keepdim = True)
            output3mean_ppma = output3_ppma - t.mean(output3_ppma, dim = 0, keepdim = True)
            output4mean_ppma = output4_ppma - t.mean(output4_ppma, dim = 0, keepdim = True)
            output5mean_ppma = output5_ppma - t.mean(output5_ppma, dim = 0, keepdim = True)
            
            normwmean_actemb_ppma["layer 0"].append(t.mean(t.norm(output0mean_ppma, dim = -1), dim = 0))
            normwmean_actemb_ppma["layer 1"].append(t.mean(t.norm(output1mean_ppma, dim = -1), dim = 0))
            normwmean_actemb_ppma["layer 2"].append(t.mean(t.norm(output2mean_ppma, dim = -1), dim = 0))
            normwmean_actemb_ppma["layer 3"].append(t.mean(t.norm(output3mean_ppma, dim = -1), dim = 0))
            normwmean_actemb_ppma["layer 4"].append(t.mean(t.norm(output4mean_ppma, dim = -1), dim = 0))
            normwmean_actemb_ppma["layer 5"].append(t.mean(t.norm(output5mean_ppma, dim = -1), dim = 0))
            
            del output0_ppma, output1_ppma, output2_ppma, output3_ppma, output4_ppma
            gc.collect()
            
        return normwmean_actemb_ppma

    def act_normwmean_ppma(self):
        
        wmean_actemb_ppma = self.act_normwmean_fn_ppma()

        
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
        normwmean_actemb_ppma["layer 5"] = np.mean(np.array(wmean_actemb_ppma["layer 5"]), axis=0)

        actlistmean_ppma = np.array([
            np.log(np.array(normwmean_actemb_ppma["layer 0"])),
            np.log(np.array(normwmean_actemb_ppma["layer 1"])),
            np.log(np.array(normwmean_actemb_ppma["layer 2"])),
            np.log(np.array(normwmean_actemb_ppma["layer 3"])),
            np.log(np.array(normwmean_actemb_ppma["layer 4"])),
            np.log(np.array(normwmean_actemb_ppma["layer 5"]))
            ])
        
        try:
            os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
        except:
            pass

        self.plotting(data=actlistmean_ppma, name = f"figures/{self.dataset_name}/{self.model_name}/activation_resid_wmean.png")

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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of residual post-mlp addn")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia14_70_mlp:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
        
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
                    output0 = self.model.gpt_neox.layers[0].mlp.output.save()
                    output1 = self.model.gpt_neox.layers[1].mlp.output.save()
                    output2 = self.model.gpt_neox.layers[2].mlp.output.save()
                    output3 = self.model.gpt_neox.layers[3].mlp.output.save()
                    output4 = self.model.gpt_neox.layers[4].mlp.output.save()
                    output5 = self.model.gpt_neox.layers[5].mlp.output.save()

                # output0.shape -> (batch_size, 128, 2048)
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
        norm_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": [],
            "layer 5": []
        }
        
        activation_embeds = self.activation_embeds_fn()
        
        
        norm_actemb["layer 0"] = np.mean(activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(activation_embeds["layer 4"], axis=0)
        norm_actemb["layer 5"] = np.mean(activation_embeds["layer 5"], axis=0)
        
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        actlist = np.array([
            np.log(np.array(norm_actemb["layer 0"])),
            np.log(np.array(norm_actemb["layer 1"])),
            np.log(np.array(norm_actemb["layer 2"])),
            np.log(np.array(norm_actemb["layer 3"])),
            np.log(np.array(norm_actemb["layer 4"])),
            np.log(np.array(norm_actemb["layer 5"]))
            # mean_acts["last layer"]
            ])
        
        try:
            os.makedirs(f"figures/{self.dataset_name}/{self.model_name}")
            os.makedirs(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_mlp.png")
        
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_mlp.pkl", "wb") as f:
            pickle.dump(actlist, f)


    def act_normwmean_fn_mlp(self):
        
        normwmean_actemb_mlp = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        for sample in tqdm(self.data):
            
            inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
                
            with self.model.trace(inputs) as tracer:
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
        
        normwmean_actemb_mlp = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        wmean_actemb_mlp = self.act_normwmean_fn_mlp()
        
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
        try:
            os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
        except:
            pass
        self.plotting(data=actlistmean_mlp, name = f"figures/{self.dataset_name}/{self.model_name}/activation_mlp_wmean.png")


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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of MLP")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia14_70_attention:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
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
                    output0 = self.model.gpt_neox.layers[0].attention.output[0].save()
                    output1 = self.model.gpt_neox.layers[1].attention.output[0].save()
                    output2 = self.model.gpt_neox.layers[2].attention.output[0].save()
                    output3 = self.model.gpt_neox.layers[3].attention.output[0].save()
                    output4 = self.model.gpt_neox.layers[4].attention.output[0].save()
                    output5 = self.model.gpt_neox.layers[5].attention.output[0].save()

                # output0.shape -> (batch_size, 128, 2048)
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
        norm_actemb = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": [],
            "layer 5": []
        }
        
        activation_embeds = self.activation_embeds_fn()
        
        
        norm_actemb["layer 0"] = np.mean(activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(activation_embeds["layer 4"], axis=0)
        norm_actemb["layer 5"] = np.mean(activation_embeds["layer 5"], axis=0)
        
        # self.actemb["last layer"] = np.linalg.norm(self.actemb["last layer"], axis=0)
        
        actlist = np.array([
            np.log(np.array(norm_actemb["layer 0"])),
            np.log(np.array(norm_actemb["layer 1"])),
            np.log(np.array(norm_actemb["layer 2"])),
            np.log(np.array(norm_actemb["layer 3"])),
            np.log(np.array(norm_actemb["layer 4"])),
            np.log(np.array(norm_actemb["layer 5"]))
            ])
        
        try:
            os.makedirs(f"figures/{self.dataset_name}/{self.model_name}")
            os.makedirs(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_attn.pkl", "wb") as f:
            pickle.dump(actlist, f)
        
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_attn.png")
        
    def act_normwmean_fn_attn(self):
        
        normwmean_actemb_attn = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        for sample in tqdm(self.data):
            
            inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
            
            with self.model.trace(inputs) as tracer:
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
        
        
        normwmean_actemb_attn = {
            "layer 0": [],
            "layer 1": [],
            "layer 2": [],
            "layer 3": [],
            "layer 4": []
        }
        
        wmean_actemb_attn = self.act_normwmean_fn_attn()
        
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
        try:
            os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
        except:
            pass
        self.plotting(data=actlistmean_attn, name = f"figures/{self.dataset_name}/{self.model_name}/activation_attn_wmean.png")


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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of Attention")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia160_resid_post_mlp_addn:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        print(model)
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
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
            "layer 11": [],
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
                    output6 = self.model.gpt_neox.layers[6].output[0].save()
                    output7 = self.model.gpt_neox.layers[7].output[0].save()
                    output8 = self.model.gpt_neox.layers[8].output[0].save()
                    output9 = self.model.gpt_neox.layers[9].output[0].save()
                    output10 = self.model.gpt_neox.layers[10].output[0].save()
                    output11 = self.model.gpt_neox.layers[11].output[0].save()
                
                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 6"].append(t.norm(output6.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 7"].append(t.norm(output7.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 8"].append(t.norm(output8.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 9"].append(t.norm(output9.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 10"].append(t.norm(output10.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 11"].append(t.norm(output11.detach().cpu(), dim = -1).squeeze(0))
                
                
        return activation_embeds

        
    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        
        activation_embeds = self.activation_embeds_fn()
        
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
        
        
        norm_actemb["layer 0"] = np.mean(np.array(activation_embeds["layer 0"]), axis=0)
        norm_actemb["layer 1"] = np.mean(np.array(activation_embeds["layer 1"]), axis=0)
        norm_actemb["layer 2"] = np.mean(np.array(activation_embeds["layer 2"]), axis=0)
        norm_actemb["layer 3"] = np.mean(np.array(activation_embeds["layer 3"]), axis=0)
        norm_actemb["layer 4"] = np.mean(np.array(activation_embeds["layer 4"]), axis=0)
        norm_actemb["layer 5"] = np.mean(np.array(activation_embeds["layer 5"]), axis=0)
        norm_actemb["layer 6"] = np.mean(np.array(activation_embeds["layer 6"]), axis=0)
        norm_actemb["layer 7"] = np.mean(np.array(activation_embeds["layer 7"]), axis=0)
        norm_actemb["layer 8"] = np.mean(np.array(activation_embeds["layer 8"]), axis=0)
        norm_actemb["layer 9"] = np.mean(np.array(activation_embeds["layer 9"]), axis=0)
        norm_actemb["layer 10"] = np.mean(np.array(activation_embeds["layer 10"]), axis=0)
        norm_actemb["layer 11"] = np.mean(np.array(activation_embeds["layer 11"]), axis=0)
        
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
            np.log(np.array(norm_actemb["layer 11"]))
            ])
            # mean_acts["last layer"]
        
        
        try:
            os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
            os.mkdir(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_resid.pkl", "wb") as f:
            pickle.dump(actlist, f)
        
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_resid.png")


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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of residual post-mlp addn")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia160_mlp:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
        
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
            for sample in tqdm(self.data):
                
                inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
                
                with self.model.trace(inputs) as tracer:
                    output0 = self.model.gpt_neox.layers[0].mlp.output.save()
                    output1 = self.model.gpt_neox.layers[1].mlp.output.save()
                    output2 = self.model.gpt_neox.layers[2].mlp.output.save()
                    output3 = self.model.gpt_neox.layers[3].mlp.output.save()
                    output4 = self.model.gpt_neox.layers[4].mlp.output.save()
                    output5 = self.model.gpt_neox.layers[5].mlp.output.save()
                    output6 = self.model.gpt_neox.layers[6].mlp.output.save()
                    output7 = self.model.gpt_neox.layers[7].mlp.output.save()
                    output8 = self.model.gpt_neox.layers[8].mlp.output.save()
                    output9 = self.model.gpt_neox.layers[9].mlp.output.save()
                    output10 = self.model.gpt_neox.layers[10].mlp.output.save()
                    output11 = self.model.gpt_neox.layers[11].mlp.output.save()

                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 6"].append(t.norm(output6.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 7"].append(t.norm(output7.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 8"].append(t.norm(output8.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 9"].append(t.norm(output9.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 10"].append(t.norm(output10.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 11"].append(t.norm(output11.detach().cpu(), dim = -1).squeeze(0))
                
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
        
        activation_embeds = self.activation_embeds_fn()
        
        
        norm_actemb["layer 0"] = np.mean(activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(activation_embeds["layer 4"], axis=0)
        norm_actemb["layer 5"] = np.mean(activation_embeds["layer 5"], axis=0)
        norm_actemb["layer 6"] = np.mean(activation_embeds["layer 6"], axis=0)
        norm_actemb["layer 7"] = np.mean(activation_embeds["layer 7"], axis=0)
        norm_actemb["layer 8"] = np.mean(activation_embeds["layer 8"], axis=0)
        norm_actemb["layer 9"] = np.mean(activation_embeds["layer 9"], axis=0)
        norm_actemb["layer 10"] = np.mean(activation_embeds["layer 10"], axis=0)
        norm_actemb["layer 11"] = np.mean(activation_embeds["layer 11"], axis=0)
        
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
            np.log(np.array(norm_actemb["layer 11"]))
            ])
            # mean_acts["last layer"]
        
        try:
            os.makedirs(f"figures/{self.dataset_name}/{self.model_name}")
            os.makedirs(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_mlp.png")
        
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_mlp.pkl", "wb") as f:
            pickle.dump(actlist, f)


 
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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of MLP")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia160_attention:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
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
            for sample in tqdm(self.data):
                
                inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
                
                with self.model.trace(inputs) as tracer:
                    output0 = self.model.gpt_neox.layers[0].attention.output[0].save()
                    output1 = self.model.gpt_neox.layers[1].attention.output[0].save()
                    output2 = self.model.gpt_neox.layers[2].attention.output[0].save()
                    output3 = self.model.gpt_neox.layers[3].attention.output[0].save()
                    output4 = self.model.gpt_neox.layers[4].attention.output[0].save()
                    output5 = self.model.gpt_neox.layers[5].attention.output[0].save()
                    output6 = self.model.gpt_neox.layers[6].attention.output[0].save()
                    output7 = self.model.gpt_neox.layers[7].attention.output[0].save()
                    output8 = self.model.gpt_neox.layers[8].attention.output[0].save()
                    output9 = self.model.gpt_neox.layers[9].attention.output[0].save()
                    output10 = self.model.gpt_neox.layers[10].attention.output[0].save()
                    output11 = self.model.gpt_neox.layers[11].attention.output[0].save()

                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 6"].append(t.norm(output6.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 7"].append(t.norm(output7.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 8"].append(t.norm(output8.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 9"].append(t.norm(output9.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 10"].append(t.norm(output10.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 11"].append(t.norm(output11.detach().cpu(), dim = -1).squeeze(0))
                
                
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
        
        activation_embeds = self.activation_embeds_fn()
        
        
        norm_actemb["layer 0"] = np.mean(activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(activation_embeds["layer 4"], axis=0)
        norm_actemb["layer 5"] = np.mean(activation_embeds["layer 5"], axis=0)
        norm_actemb["layer 6"] = np.mean(activation_embeds["layer 6"], axis=0)
        norm_actemb["layer 7"] = np.mean(activation_embeds["layer 7"], axis=0)
        norm_actemb["layer 8"] = np.mean(activation_embeds["layer 8"], axis=0)
        norm_actemb["layer 9"] = np.mean(activation_embeds["layer 9"], axis=0)
        norm_actemb["layer 10"] = np.mean(activation_embeds["layer 10"], axis=0)
        norm_actemb["layer 11"] = np.mean(activation_embeds["layer 11"], axis=0)
        
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
            np.log(np.array(norm_actemb["layer 11"]))
            ])
        
        try:
            os.makedirs(f"figures/{self.dataset_name}/{self.model_name}")
            os.makedirs(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_attn.pkl", "wb") as f:
            pickle.dump(actlist, f)
        
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_attn.png")
        
    
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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of Attention")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia410_1_4_resid_post_mlp_addn:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        print(model)
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
            "layer 16": [],
            "layer 17": [],
            "layer 18": [],
            "layer 19": [],
            "layer 20": [],
            "layer 21": [],
            "layer 22": [],
            "layer 23": [],
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
                    output6 = self.model.gpt_neox.layers[6].output[0].save()
                    output7 = self.model.gpt_neox.layers[7].output[0].save()
                    output8 = self.model.gpt_neox.layers[8].output[0].save()
                    output9 = self.model.gpt_neox.layers[9].output[0].save()
                    output10 = self.model.gpt_neox.layers[10].output[0].save()
                    output11 = self.model.gpt_neox.layers[11].output[0].save()
                    output12 = self.model.gpt_neox.layers[12].output[0].save()
                    output13 = self.model.gpt_neox.layers[13].output[0].save()
                    output14 = self.model.gpt_neox.layers[14].output[0].save()
                    output15 = self.model.gpt_neox.layers[15].output[0].save()
                    output16 = self.model.gpt_neox.layers[16].output[0].save()
                    output17 = self.model.gpt_neox.layers[17].output[0].save()
                    output18 = self.model.gpt_neox.layers[18].output[0].save()
                    output19 = self.model.gpt_neox.layers[19].output[0].save()
                    output20 = self.model.gpt_neox.layers[20].output[0].save()
                    output21 = self.model.gpt_neox.layers[21].output[0].save()
                    output22 = self.model.gpt_neox.layers[22].output[0].save()
                    output23 = self.model.gpt_neox.layers[23].output[0].save()
                    
                
                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 6"].append(t.norm(output6.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 7"].append(t.norm(output7.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 8"].append(t.norm(output8.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 9"].append(t.norm(output9.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 10"].append(t.norm(output10.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 11"].append(t.norm(output11.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 12"].append(t.norm(output12.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 13"].append(t.norm(output13.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 14"].append(t.norm(output14.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 15"].append(t.norm(output15.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 16"].append(t.norm(output16.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 17"].append(t.norm(output17.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 18"].append(t.norm(output18.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 19"].append(t.norm(output19.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 20"].append(t.norm(output20.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 21"].append(t.norm(output21.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 22"].append(t.norm(output22.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 23"].append(t.norm(output23.detach().cpu(), dim = -1).squeeze(0))
                
                
        return activation_embeds

        
    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        
        activation_embeds = self.activation_embeds_fn()
        
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
            "layer 16": [],
            "layer 17": [],
            "layer 18": [],
            "layer 19": [],
            "layer 20": [],
            "layer 21": [],
            "layer 22": [],
            "layer 23": []
        }
        
        
        norm_actemb["layer 0"] = np.mean(np.array(activation_embeds["layer 0"]), axis=0)
        norm_actemb["layer 1"] = np.mean(np.array(activation_embeds["layer 1"]), axis=0)
        norm_actemb["layer 2"] = np.mean(np.array(activation_embeds["layer 2"]), axis=0)
        norm_actemb["layer 3"] = np.mean(np.array(activation_embeds["layer 3"]), axis=0)
        norm_actemb["layer 4"] = np.mean(np.array(activation_embeds["layer 4"]), axis=0)
        norm_actemb["layer 5"] = np.mean(np.array(activation_embeds["layer 5"]), axis=0)
        norm_actemb["layer 6"] = np.mean(np.array(activation_embeds["layer 6"]), axis=0)
        norm_actemb["layer 7"] = np.mean(np.array(activation_embeds["layer 7"]), axis=0)
        norm_actemb["layer 8"] = np.mean(np.array(activation_embeds["layer 8"]), axis=0)
        norm_actemb["layer 9"] = np.mean(np.array(activation_embeds["layer 9"]), axis=0)
        norm_actemb["layer 10"] = np.mean(np.array(activation_embeds["layer 10"]), axis=0)
        norm_actemb["layer 11"] = np.mean(np.array(activation_embeds["layer 11"]), axis=0)
        norm_actemb["layer 12"] = np.mean(np.array(activation_embeds["layer 12"]), axis=0)
        norm_actemb["layer 13"] = np.mean(np.array(activation_embeds["layer 13"]), axis=0)
        norm_actemb["layer 14"] = np.mean(np.array(activation_embeds["layer 14"]), axis=0)
        norm_actemb["layer 15"] = np.mean(np.array(activation_embeds["layer 15"]), axis=0)
        norm_actemb["layer 16"] = np.mean(np.array(activation_embeds["layer 16"]), axis=0)
        norm_actemb["layer 17"] = np.mean(np.array(activation_embeds["layer 17"]), axis=0)
        norm_actemb["layer 18"] = np.mean(np.array(activation_embeds["layer 18"]), axis=0)
        norm_actemb["layer 19"] = np.mean(np.array(activation_embeds["layer 19"]), axis=0)
        norm_actemb["layer 20"] = np.mean(np.array(activation_embeds["layer 20"]), axis=0)
        norm_actemb["layer 21"] = np.mean(np.array(activation_embeds["layer 21"]), axis=0)
        norm_actemb["layer 22"] = np.mean(np.array(activation_embeds["layer 22"]), axis=0)
        norm_actemb["layer 23"] = np.mean(np.array(activation_embeds["layer 23"]), axis=0)
        
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
            np.log(np.array(norm_actemb["layer 12"])),
            np.log(np.array(norm_actemb["layer 13"])),
            np.log(np.array(norm_actemb["layer 14"])),
            np.log(np.array(norm_actemb["layer 15"])),
            np.log(np.array(norm_actemb["layer 16"])),
            np.log(np.array(norm_actemb["layer 17"])),
            np.log(np.array(norm_actemb["layer 18"])),
            np.log(np.array(norm_actemb["layer 19"])),
            np.log(np.array(norm_actemb["layer 20"])),
            np.log(np.array(norm_actemb["layer 21"])),
            np.log(np.array(norm_actemb["layer 22"])),
            np.log(np.array(norm_actemb["layer 23"]))
            ])
            # mean_acts["last layer"]
        
        
        try:
            os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
            os.mkdir(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_resid.pkl", "wb") as f:
            pickle.dump(actlist, f)
        
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_resid.png")


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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of residual post-mlp addn")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia410_1_4_mlp:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
        
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
            "layer 16": [],
            "layer 17": [],
            "layer 18": [],
            "layer 19": [],
            "layer 20": [],
            "layer 21": [],
            "layer 22": [],
            "layer 23": []
        }
        
        with t.no_grad():
            for sample in tqdm(self.data):
                
                inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
                
                with self.model.trace(inputs) as tracer:
                    output0 = self.model.gpt_neox.layers[0].mlp.output.save()
                    output1 = self.model.gpt_neox.layers[1].mlp.output.save()
                    output2 = self.model.gpt_neox.layers[2].mlp.output.save()
                    output3 = self.model.gpt_neox.layers[3].mlp.output.save()
                    output4 = self.model.gpt_neox.layers[4].mlp.output.save()
                    output5 = self.model.gpt_neox.layers[5].mlp.output.save()
                    output6 = self.model.gpt_neox.layers[6].mlp.output.save()
                    output7 = self.model.gpt_neox.layers[7].mlp.output.save()
                    output8 = self.model.gpt_neox.layers[8].mlp.output.save()
                    output9 = self.model.gpt_neox.layers[9].mlp.output.save()
                    output10 = self.model.gpt_neox.layers[10].mlp.output.save()
                    output11 = self.model.gpt_neox.layers[11].mlp.output.save()
                    output12 = self.model.gpt_neox.layers[12].mlp.output.save()
                    output13 = self.model.gpt_neox.layers[13].mlp.output.save()
                    output14 = self.model.gpt_neox.layers[14].mlp.output.save()
                    output15 = self.model.gpt_neox.layers[15].mlp.output.save()
                    output16 = self.model.gpt_neox.layers[16].mlp.output.save()
                    output17 = self.model.gpt_neox.layers[17].mlp.output.save()
                    output18 = self.model.gpt_neox.layers[18].mlp.output.save()
                    output19 = self.model.gpt_neox.layers[19].mlp.output.save()
                    output20 = self.model.gpt_neox.layers[20].mlp.output.save()
                    output21 = self.model.gpt_neox.layers[21].mlp.output.save()
                    output22 = self.model.gpt_neox.layers[22].mlp.output.save()
                    output23 = self.model.gpt_neox.layers[23].mlp.output.save()
                    

                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 6"].append(t.norm(output6.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 7"].append(t.norm(output7.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 8"].append(t.norm(output8.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 9"].append(t.norm(output9.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 10"].append(t.norm(output10.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 11"].append(t.norm(output11.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 12"].append(t.norm(output12.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 13"].append(t.norm(output13.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 14"].append(t.norm(output14.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 15"].append(t.norm(output15.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 16"].append(t.norm(output16.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 17"].append(t.norm(output17.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 18"].append(t.norm(output18.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 19"].append(t.norm(output19.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 20"].append(t.norm(output20.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 21"].append(t.norm(output21.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 22"].append(t.norm(output22.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 23"].append(t.norm(output23.detach().cpu(), dim = -1).squeeze(0))
                
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
            "layer 16": [],
            "layer 17": [],
            "layer 18": [],
            "layer 19": [],
            "layer 20": [],
            "layer 21": [],
            "layer 22": [],
            "layer 23": []
        }
        
        activation_embeds = self.activation_embeds_fn()
        
        
        norm_actemb["layer 0"] = np.mean(activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(activation_embeds["layer 4"], axis=0)
        norm_actemb["layer 5"] = np.mean(activation_embeds["layer 5"], axis=0)
        norm_actemb["layer 6"] = np.mean(activation_embeds["layer 6"], axis=0)
        norm_actemb["layer 7"] = np.mean(activation_embeds["layer 7"], axis=0)
        norm_actemb["layer 8"] = np.mean(activation_embeds["layer 8"], axis=0)
        norm_actemb["layer 9"] = np.mean(activation_embeds["layer 9"], axis=0)
        norm_actemb["layer 10"] = np.mean(activation_embeds["layer 10"], axis=0)
        norm_actemb["layer 11"] = np.mean(activation_embeds["layer 11"], axis=0)
        norm_actemb["layer 12"] = np.mean(activation_embeds["layer 12"], axis=0)
        norm_actemb["layer 13"] = np.mean(activation_embeds["layer 13"], axis=0)
        norm_actemb["layer 14"] = np.mean(activation_embeds["layer 14"], axis=0)
        norm_actemb["layer 15"] = np.mean(activation_embeds["layer 15"], axis=0)
        norm_actemb["layer 16"] = np.mean(activation_embeds["layer 16"], axis=0)
        norm_actemb["layer 17"] = np.mean(activation_embeds["layer 17"], axis=0)
        norm_actemb["layer 18"] = np.mean(activation_embeds["layer 18"], axis=0)
        norm_actemb["layer 19"] = np.mean(activation_embeds["layer 19"], axis=0)
        norm_actemb["layer 20"] = np.mean(activation_embeds["layer 20"], axis=0)
        norm_actemb["layer 21"] = np.mean(activation_embeds["layer 21"], axis=0)
        norm_actemb["layer 22"] = np.mean(activation_embeds["layer 22"], axis=0)
        norm_actemb["layer 23"] = np.mean(activation_embeds["layer 23"], axis=0)
        
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
            np.log(np.array(norm_actemb["layer 12"])),
            np.log(np.array(norm_actemb["layer 13"])),
            np.log(np.array(norm_actemb["layer 14"])),
            np.log(np.array(norm_actemb["layer 15"])),
            np.log(np.array(norm_actemb["layer 16"])),
            np.log(np.array(norm_actemb["layer 17"])),
            np.log(np.array(norm_actemb["layer 18"])),
            np.log(np.array(norm_actemb["layer 19"])),
            np.log(np.array(norm_actemb["layer 20"])),
            np.log(np.array(norm_actemb["layer 21"])),
            np.log(np.array(norm_actemb["layer 22"])),
            np.log(np.array(norm_actemb["layer 23"]))
            ])
            # mean_acts["last layer"]
        
        try:
            os.makedirs(f"figures/{self.dataset_name}/{self.model_name}")
            os.makedirs(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_mlp.png")
        
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_mlp.pkl", "wb") as f:
            pickle.dump(actlist, f)


 
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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of MLP")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia410_1_4_attention:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
            "layer 16": [],
            "layer 17": [],
            "layer 18": [],
            "layer 19": [],
            "layer 20": [],
            "layer 21": [],
            "layer 22": [],
            "layer 23": []
        }
        
        with t.no_grad():
            for sample in tqdm(self.data):
                
                inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
                
                with self.model.trace(inputs) as tracer:
                    output0 = self.model.gpt_neox.layers[0].attention.output[0].save()
                    output1 = self.model.gpt_neox.layers[1].attention.output[0].save()
                    output2 = self.model.gpt_neox.layers[2].attention.output[0].save()
                    output3 = self.model.gpt_neox.layers[3].attention.output[0].save()
                    output4 = self.model.gpt_neox.layers[4].attention.output[0].save()
                    output5 = self.model.gpt_neox.layers[5].attention.output[0].save()
                    output6 = self.model.gpt_neox.layers[6].attention.output[0].save()
                    output7 = self.model.gpt_neox.layers[7].attention.output[0].save()
                    output8 = self.model.gpt_neox.layers[8].attention.output[0].save()
                    output9 = self.model.gpt_neox.layers[9].attention.output[0].save()
                    output10 = self.model.gpt_neox.layers[10].attention.output[0].save()
                    output11 = self.model.gpt_neox.layers[11].attention.output[0].save()
                    output12 = self.model.gpt_neox.layers[12].attention.output[0].save()
                    output13 = self.model.gpt_neox.layers[13].attention.output[0].save()
                    output14 = self.model.gpt_neox.layers[14].attention.output[0].save()
                    output15 = self.model.gpt_neox.layers[15].attention.output[0].save()
                    output16 = self.model.gpt_neox.layers[16].attention.output[0].save()
                    output17 = self.model.gpt_neox.layers[17].attention.output[0].save()
                    output18 = self.model.gpt_neox.layers[18].attention.output[0].save()
                    output19 = self.model.gpt_neox.layers[19].attention.output[0].save()
                    output20 = self.model.gpt_neox.layers[20].attention.output[0].save()
                    output21 = self.model.gpt_neox.layers[21].attention.output[0].save()
                    output22 = self.model.gpt_neox.layers[22].attention.output[0].save()
                    output23 = self.model.gpt_neox.layers[23].attention.output[0].save()

                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 6"].append(t.norm(output6.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 7"].append(t.norm(output7.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 8"].append(t.norm(output8.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 9"].append(t.norm(output9.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 10"].append(t.norm(output10.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 11"].append(t.norm(output11.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 12"].append(t.norm(output12.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 13"].append(t.norm(output13.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 14"].append(t.norm(output14.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 15"].append(t.norm(output15.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 16"].append(t.norm(output16.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 17"].append(t.norm(output17.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 18"].append(t.norm(output18.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 19"].append(t.norm(output19.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 20"].append(t.norm(output20.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 21"].append(t.norm(output21.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 22"].append(t.norm(output22.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 23"].append(t.norm(output23.detach().cpu(), dim = -1).squeeze(0))
                
                
                
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
            "layer 16": [],
            "layer 17": [],
            "layer 18": [],
            "layer 19": [],
            "layer 20": [],
            "layer 21": [],
            "layer 22": [],
            "layer 23": []
        }
        
        activation_embeds = self.activation_embeds_fn()
        
        
        norm_actemb["layer 0"] = np.mean(activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(activation_embeds["layer 4"], axis=0)
        norm_actemb["layer 5"] = np.mean(activation_embeds["layer 5"], axis=0)
        norm_actemb["layer 6"] = np.mean(activation_embeds["layer 6"], axis=0)
        norm_actemb["layer 7"] = np.mean(activation_embeds["layer 7"], axis=0)
        norm_actemb["layer 8"] = np.mean(activation_embeds["layer 8"], axis=0)
        norm_actemb["layer 9"] = np.mean(activation_embeds["layer 9"], axis=0)
        norm_actemb["layer 10"] = np.mean(activation_embeds["layer 10"], axis=0)
        norm_actemb["layer 11"] = np.mean(activation_embeds["layer 11"], axis=0)
        norm_actemb["layer 12"] = np.mean(activation_embeds["layer 12"], axis=0)
        norm_actemb["layer 13"] = np.mean(activation_embeds["layer 13"], axis=0)
        norm_actemb["layer 14"] = np.mean(activation_embeds["layer 14"], axis=0)
        norm_actemb["layer 15"] = np.mean(activation_embeds["layer 15"], axis=0)
        norm_actemb["layer 16"] = np.mean(activation_embeds["layer 16"], axis=0)
        norm_actemb["layer 17"] = np.mean(activation_embeds["layer 17"], axis=0)
        norm_actemb["layer 18"] = np.mean(activation_embeds["layer 18"], axis=0)
        norm_actemb["layer 19"] = np.mean(activation_embeds["layer 19"], axis=0)
        norm_actemb["layer 20"] = np.mean(activation_embeds["layer 20"], axis=0)
        norm_actemb["layer 21"] = np.mean(activation_embeds["layer 21"], axis=0)
        norm_actemb["layer 22"] = np.mean(activation_embeds["layer 22"], axis=0)
        norm_actemb["layer 23"] = np.mean(activation_embeds["layer 23"], axis=0)
        
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
            np.log(np.array(norm_actemb["layer 12"])),
            np.log(np.array(norm_actemb["layer 13"])),
            np.log(np.array(norm_actemb["layer 14"])),
            np.log(np.array(norm_actemb["layer 15"])),
            np.log(np.array(norm_actemb["layer 16"])),
            np.log(np.array(norm_actemb["layer 17"])),
            np.log(np.array(norm_actemb["layer 18"])),
            np.log(np.array(norm_actemb["layer 19"])),
            np.log(np.array(norm_actemb["layer 20"])),
            np.log(np.array(norm_actemb["layer 21"])),
            np.log(np.array(norm_actemb["layer 22"])),
            np.log(np.array(norm_actemb["layer 23"]))
            ])
        
        try:
            os.makedirs(f"figures/{self.dataset_name}/{self.model_name}")
            os.makedirs(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_attn.pkl", "wb") as f:
            pickle.dump(actlist, f)
        
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_attn.png")
        
    
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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of Attention")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia_1_resid_post_mlp_addn:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        print(model)
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
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
                    output6 = self.model.gpt_neox.layers[6].output[0].save()
                    output7 = self.model.gpt_neox.layers[7].output[0].save()
                    output8 = self.model.gpt_neox.layers[8].output[0].save()
                    output9 = self.model.gpt_neox.layers[9].output[0].save()
                    output10 = self.model.gpt_neox.layers[10].output[0].save()
                    output11 = self.model.gpt_neox.layers[11].output[0].save()
                    output12 = self.model.gpt_neox.layers[12].output[0].save()
                    output13 = self.model.gpt_neox.layers[13].output[0].save()
                    output14 = self.model.gpt_neox.layers[14].output[0].save()
                    output15 = self.model.gpt_neox.layers[15].output[0].save()

                
                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 6"].append(t.norm(output6.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 7"].append(t.norm(output7.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 8"].append(t.norm(output8.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 9"].append(t.norm(output9.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 10"].append(t.norm(output10.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 11"].append(t.norm(output11.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 12"].append(t.norm(output12.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 13"].append(t.norm(output13.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 14"].append(t.norm(output14.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 15"].append(t.norm(output15.detach().cpu(), dim = -1).squeeze(0))
                
                
        return activation_embeds

        
    def norm(self):
        
        # Additional norm calculations for nested structures
        # assert np.array(self.actemb["layer 0"]).shape[1] == 128
        
        activation_embeds = self.activation_embeds_fn()
        
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
        }
        
        
        norm_actemb["layer 0"] = np.mean(np.array(activation_embeds["layer 0"]), axis=0)
        norm_actemb["layer 1"] = np.mean(np.array(activation_embeds["layer 1"]), axis=0)
        norm_actemb["layer 2"] = np.mean(np.array(activation_embeds["layer 2"]), axis=0)
        norm_actemb["layer 3"] = np.mean(np.array(activation_embeds["layer 3"]), axis=0)
        norm_actemb["layer 4"] = np.mean(np.array(activation_embeds["layer 4"]), axis=0)
        norm_actemb["layer 5"] = np.mean(np.array(activation_embeds["layer 5"]), axis=0)
        norm_actemb["layer 6"] = np.mean(np.array(activation_embeds["layer 6"]), axis=0)
        norm_actemb["layer 7"] = np.mean(np.array(activation_embeds["layer 7"]), axis=0)
        norm_actemb["layer 8"] = np.mean(np.array(activation_embeds["layer 8"]), axis=0)
        norm_actemb["layer 9"] = np.mean(np.array(activation_embeds["layer 9"]), axis=0)
        norm_actemb["layer 10"] = np.mean(np.array(activation_embeds["layer 10"]), axis=0)
        norm_actemb["layer 11"] = np.mean(np.array(activation_embeds["layer 11"]), axis=0)
        norm_actemb["layer 12"] = np.mean(np.array(activation_embeds["layer 12"]), axis=0)
        norm_actemb["layer 13"] = np.mean(np.array(activation_embeds["layer 13"]), axis=0)
        norm_actemb["layer 14"] = np.mean(np.array(activation_embeds["layer 14"]), axis=0)
        norm_actemb["layer 15"] = np.mean(np.array(activation_embeds["layer 15"]), axis=0)
        
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
            np.log(np.array(norm_actemb["layer 12"])),
            np.log(np.array(norm_actemb["layer 13"])),
            np.log(np.array(norm_actemb["layer 14"])),
            np.log(np.array(norm_actemb["layer 15"])),
            ])
            # mean_acts["last layer"]
        
        
        try:
            os.mkdir(f"figures/{self.dataset_name}/{self.model_name}")
            os.mkdir(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_resid.pkl", "wb") as f:
            pickle.dump(actlist, f)
        
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_resid.png")


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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of residual post-mlp addn")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia_1_mlp:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
        
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
        }
        
        with t.no_grad():
            for sample in tqdm(self.data):
                
                inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
                
                with self.model.trace(inputs) as tracer:
                    output0 = self.model.gpt_neox.layers[0].mlp.output.save()
                    output1 = self.model.gpt_neox.layers[1].mlp.output.save()
                    output2 = self.model.gpt_neox.layers[2].mlp.output.save()
                    output3 = self.model.gpt_neox.layers[3].mlp.output.save()
                    output4 = self.model.gpt_neox.layers[4].mlp.output.save()
                    output5 = self.model.gpt_neox.layers[5].mlp.output.save()
                    output6 = self.model.gpt_neox.layers[6].mlp.output.save()
                    output7 = self.model.gpt_neox.layers[7].mlp.output.save()
                    output8 = self.model.gpt_neox.layers[8].mlp.output.save()
                    output9 = self.model.gpt_neox.layers[9].mlp.output.save()
                    output10 = self.model.gpt_neox.layers[10].mlp.output.save()
                    output11 = self.model.gpt_neox.layers[11].mlp.output.save()
                    output12 = self.model.gpt_neox.layers[12].mlp.output.save()
                    output13 = self.model.gpt_neox.layers[13].mlp.output.save()
                    output14 = self.model.gpt_neox.layers[14].mlp.output.save()
                    output15 = self.model.gpt_neox.layers[15].mlp.output.save()

                    

                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 6"].append(t.norm(output6.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 7"].append(t.norm(output7.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 8"].append(t.norm(output8.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 9"].append(t.norm(output9.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 10"].append(t.norm(output10.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 11"].append(t.norm(output11.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 12"].append(t.norm(output12.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 13"].append(t.norm(output13.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 14"].append(t.norm(output14.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 15"].append(t.norm(output15.detach().cpu(), dim = -1).squeeze(0))
                
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
        }
        
        activation_embeds = self.activation_embeds_fn()
        
        
        norm_actemb["layer 0"] = np.mean(activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(activation_embeds["layer 4"], axis=0)
        norm_actemb["layer 5"] = np.mean(activation_embeds["layer 5"], axis=0)
        norm_actemb["layer 6"] = np.mean(activation_embeds["layer 6"], axis=0)
        norm_actemb["layer 7"] = np.mean(activation_embeds["layer 7"], axis=0)
        norm_actemb["layer 8"] = np.mean(activation_embeds["layer 8"], axis=0)
        norm_actemb["layer 9"] = np.mean(activation_embeds["layer 9"], axis=0)
        norm_actemb["layer 10"] = np.mean(activation_embeds["layer 10"], axis=0)
        norm_actemb["layer 11"] = np.mean(activation_embeds["layer 11"], axis=0)
        norm_actemb["layer 12"] = np.mean(activation_embeds["layer 12"], axis=0)
        norm_actemb["layer 13"] = np.mean(activation_embeds["layer 13"], axis=0)
        norm_actemb["layer 14"] = np.mean(activation_embeds["layer 14"], axis=0)
        norm_actemb["layer 15"] = np.mean(activation_embeds["layer 15"], axis=0)
        
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
            np.log(np.array(norm_actemb["layer 12"])),
            np.log(np.array(norm_actemb["layer 13"])),
            np.log(np.array(norm_actemb["layer 14"])),
            np.log(np.array(norm_actemb["layer 15"])),
            ])
            # mean_acts["last layer"]
        
        try:
            os.makedirs(f"figures/{self.dataset_name}/{self.model_name}")
            os.makedirs(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_mlp.png")
        
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_mlp.pkl", "wb") as f:
            pickle.dump(actlist, f)


 
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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of MLP")

        # Show the heatmap
        plt.savefig(name)
        plt.close()


class act_pythia_1_attention:

    def __init__(
        self, 
        data,
        model,
        model_name,
        dataset_name):
        
        self.data = data
        self.model = model
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.tokenizer = self.model.tokenizer
        
        if self.dataset_name == "tinystories":
            self.max_length = 145
        elif self.dataset_name == "summarisation":
            self.max_length = 340
        elif self.dataset_name == "alpaca":
            self.max_length = 10
        
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
        }
        
        with t.no_grad():
            for sample in tqdm(self.data):
                
                inputs = self.tokenizer(sample, return_tensors="pt", padding='max_length', max_length=self.max_length, truncation=True)
                
                with self.model.trace(inputs) as tracer:
                    output0 = self.model.gpt_neox.layers[0].attention.output[0].save()
                    output1 = self.model.gpt_neox.layers[1].attention.output[0].save()
                    output2 = self.model.gpt_neox.layers[2].attention.output[0].save()
                    output3 = self.model.gpt_neox.layers[3].attention.output[0].save()
                    output4 = self.model.gpt_neox.layers[4].attention.output[0].save()
                    output5 = self.model.gpt_neox.layers[5].attention.output[0].save()
                    output6 = self.model.gpt_neox.layers[6].attention.output[0].save()
                    output7 = self.model.gpt_neox.layers[7].attention.output[0].save()
                    output8 = self.model.gpt_neox.layers[8].attention.output[0].save()
                    output9 = self.model.gpt_neox.layers[9].attention.output[0].save()
                    output10 = self.model.gpt_neox.layers[10].attention.output[0].save()
                    output11 = self.model.gpt_neox.layers[11].attention.output[0].save()
                    output12 = self.model.gpt_neox.layers[12].attention.output[0].save()
                    output13 = self.model.gpt_neox.layers[13].attention.output[0].save()
                    output14 = self.model.gpt_neox.layers[14].attention.output[0].save()
                    output15 = self.model.gpt_neox.layers[15].attention.output[0].save()

                # output0.shape -> (batch_size, 128, 2048)
                activation_embeds["layer 0"].append(t.norm(output0.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 1"].append(t.norm(output1.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 2"].append(t.norm(output2.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 3"].append(t.norm(output3.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 4"].append(t.norm(output4.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 5"].append(t.norm(output5.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 6"].append(t.norm(output6.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 7"].append(t.norm(output7.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 8"].append(t.norm(output8.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 9"].append(t.norm(output9.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 10"].append(t.norm(output10.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 11"].append(t.norm(output11.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 12"].append(t.norm(output12.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 13"].append(t.norm(output13.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 14"].append(t.norm(output14.detach().cpu(), dim = -1).squeeze(0))
                activation_embeds["layer 15"].append(t.norm(output15.detach().cpu(), dim = -1).squeeze(0))

                
                
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
            "layer 11": [],
            "layer 12": [],
            "layer 13": [],
            "layer 14": [],
            "layer 15": [],
        }
        
        activation_embeds = self.activation_embeds_fn()
        
        
        norm_actemb["layer 0"] = np.mean(activation_embeds["layer 0"], axis=0)
        norm_actemb["layer 1"] = np.mean(activation_embeds["layer 1"], axis=0)
        norm_actemb["layer 2"] = np.mean(activation_embeds["layer 2"], axis=0)
        norm_actemb["layer 3"] = np.mean(activation_embeds["layer 3"], axis=0)
        norm_actemb["layer 4"] = np.mean(activation_embeds["layer 4"], axis=0)
        norm_actemb["layer 5"] = np.mean(activation_embeds["layer 5"], axis=0)
        norm_actemb["layer 6"] = np.mean(activation_embeds["layer 6"], axis=0)
        norm_actemb["layer 7"] = np.mean(activation_embeds["layer 7"], axis=0)
        norm_actemb["layer 8"] = np.mean(activation_embeds["layer 8"], axis=0)
        norm_actemb["layer 9"] = np.mean(activation_embeds["layer 9"], axis=0)
        norm_actemb["layer 10"] = np.mean(activation_embeds["layer 10"], axis=0)
        norm_actemb["layer 11"] = np.mean(activation_embeds["layer 11"], axis=0)
        norm_actemb["layer 12"] = np.mean(activation_embeds["layer 12"], axis=0)
        norm_actemb["layer 13"] = np.mean(activation_embeds["layer 13"], axis=0)
        norm_actemb["layer 14"] = np.mean(activation_embeds["layer 14"], axis=0)
        norm_actemb["layer 15"] = np.mean(activation_embeds["layer 15"], axis=0)
        
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
            np.log(np.array(norm_actemb["layer 12"])),
            np.log(np.array(norm_actemb["layer 13"])),
            np.log(np.array(norm_actemb["layer 14"])),
            np.log(np.array(norm_actemb["layer 15"])),
            ])
        
        try:
            os.makedirs(f"figures/{self.dataset_name}/{self.model_name}")
            os.makedirs(f"data/{self.dataset_name}/{self.model_name}")
        except:
            pass
        output_dir = f"data/{self.dataset_name}/{self.model_name}"
        os.makedirs(output_dir, exist_ok=True)
        with open(f"data/{self.dataset_name}/{self.model_name}/activation_attn.pkl", "wb") as f:
            pickle.dump(actlist, f)
        
        self.plotting(data=actlist, name = f"figures/{self.dataset_name}/{self.model_name}/activation_attn.png")
        
    
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
        plt.title(f"[{self.model_name}-{self.dataset_name}]:Activation of Attention")

        # Show the heatmap
        plt.savefig(name)
        plt.close()

