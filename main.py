from imports import *
from dataset import *
from activation import *
from grad import *
from counterfactual_activation import *

class config:
    
    def __init__(self, device, model_name, data_name):
        self.device = device
        self.model_name = model_name
        self.data_name = data_name


    def model_selection(self):
        if self.model_name == "Pythia14m":
            model = LanguageModel("EleutherAI/pythia-14m", device_map=self.device)
        elif self.model_name == "Pythia70m":
            model = LanguageModel("EleutherAI/pythia-70m", device_map=self.device)
        elif self.model_name == "Pythia160m":
            model = LanguageModel("EleutherAI/pythia-160m", device_map=self.device)
        elif self.model_name == "Pythia410m":
            model = LanguageModel("EleutherAI/pythia-410m", device_map=self.device)
        elif self.model_name == "Pythia1b":
            model = LanguageModel("EleutherAI/pythia-1b", device_map=self.device)
        elif self.model_name == "Pythia1.4b":
            model = LanguageModel("EleutherAI/pythia-1.4b", device_map=self.device)
        elif self.model_name == "Pythia2.8b":
            model = LanguageModel("EleutherAI/pythia-2.8b", device_map=self.device)
        elif self.model_name == "Pythia6.9b":
            model = LanguageModel("EleutherAI/pythia-6.9b", device_map=self.device)
        elif self.model_name == "Pythia12b":
            model = LanguageModel("EleutherAI/pythia-12b", device_map=self.device)
        
        return model

    def data_selection(self):
        
        if self.data_name == "tinystories":
            pre_data = load_dataset("roneneldan/TinyStories")
            data = pre_data["validation"]['text'] # as we are experimenting on the validation dataset. 
            final_data = self.process_data_selection(data)
        
        elif self.data_name == "alpaca":
            pre_data_alpaca = load_dataset("tatsu-lab/alpaca")
            data = pre_data_alpaca["train"]["instruction"]
            final_data = self.process_data_selection(data)
        
        elif self.data_name == "summarisation":
            data_summarisation = load_dataset("Lots-of-LoRAs/task298_storycloze_correct_end_classification", name="default")
            new_data = data_summarisation["train"]["input"]
            final_data = self.process_data_selection(new_data)
        
        return final_data
    
    def process_data_selection(self, data):
        
        '''
        The purpose of this function is to select 500 samples from each dataset of their average length
        - TinyStories: 177
        - Alpaca: 50
        - Summarisation: 1100
        '''
        
        if self.data_name == "tinystories":
            shuffled_text = random.sample(data, len(data))
            new_data = []
            for sent in shuffled_text:
                if len(sent.split()) == 143 or len(sent.split()) == 142:
                    new_data.append(sent)
                if len(new_data) == 400:
                    break
                
        elif self.data_name == "alpaca":
            shuffled_text = random.sample(data, len(data))
            new_data = []
            for sent in shuffled_text:
                if len(sent.split()) == 7:
                    new_data.append(sent)
                if len(new_data) == 400:
                    break
        elif self.data_name == "summarisation":
            shuffled_text = random.sample(data, len(data))
            new_data = []
            for sent in shuffled_text:
                if len(sent.split()) > 310 or len(sent.split()) < 320:
                    new_data.append(sent)
                if len(new_data) == 400:
                    break
        
        assert len(new_data) == 400
        
        return new_data

if __name__ == "__main__":
    
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type = str, default="Pythia70m")
    parser.add_argument("--batch_size", type=int, default="16")
    parser.add_argument("--data", type = str, default = "tinystories")
    parser.add_argument("--device", type = str, default = "mps")
    parser.add_argument("--task", type = str, required=True)
    parser.add_argument("--layer_type", type = str, required=True)
    
    args = parser.parse_args()
    
    model_name = args.model_name
    batch_size = args.batch_size
    dataset_used = args.data
    
    configuration = config(device = args.device, model_name = model_name, data_name = dataset_used)
    data = configuration.data_selection()
    model = configuration.model_selection()
    
    

    # Computing the Attention Norms
    
    if args.model_name == "Pythia16m" or args.model_name == "Pythia70m" and args.task == "activations":
            
        activation_resid= act_pythia14_70_resid_post_mlp_addn(data, model, model_name, dataset_used)
        activation_resid.norm()
        

        activation_mlp = act_pythia14_70_mlp(data, model, model_name, dataset_used)
        activation_mlp.norm()
        
        
        activation_attn = act_pythia14_70_attention(data, model, model_name, dataset_used)
        activation_attn.norm()
    
    
    elif args.model_name == "Pythia410m" or args.model_name == "Pythia1.4b" and args.task == "activations":
            
        activation_resid= act_pythia410_1_4_resid_post_mlp_addn(data, model, model_name, dataset_used)
        activation_resid.norm()
        

        activation_mlp = act_pythia410_1_4_mlp(data, model, model_name, dataset_used)
        activation_mlp.norm()
        
        
        activation_attn = act_pythia410_1_4_attention(data, model, model_name, dataset_used)
        activation_attn.norm()
    
    
    elif args.model_name == "Pythia160m" and args.task == "activations":
            
        activation_resid= act_pythia160_resid_post_mlp_addn(data, model, model_name, dataset_used)
        activation_resid.norm()
        

        activation_mlp = act_pythia160_mlp(data, model, model_name, dataset_used)
        activation_mlp.norm()
        
        
        activation_attn = act_pythia160_attention(data, model, model_name, dataset_used)
        activation_attn.norm()


    elif args.model_name == "Pythia1b" and args.task == "activations":
            
        activation_resid= act_pythia_1_resid_post_mlp_addn(data, model, model_name, dataset_used)
        activation_resid.norm()
        

        activation_mlp = act_pythia_1_mlp(data, model, model_name, dataset_used)
        activation_mlp.norm()
        
        
        activation_attn = act_pythia_1_attention(data, model, model_name, dataset_used)
        activation_attn.norm()
    
    elif args.model_name == "Pythia70m" or args.model_name == "Pythia14m" and args.task == "counterfactual activations":
        counterfactual_activations = counterfactual_activation_1470m(
            data = data,
            model= model,
            model_name = model_name, 
            dataset_name = args.data, 
            layer_type = args.layer_type)
        
        counterfactual_activations.norm()
    
    
    # Computing the attention gradients

    # grad_mlp = Gradient_MLP(data, args.device, dataset_used, model_name)
    # grad_mlp.forward()
    
    