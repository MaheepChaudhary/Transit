from imports import *
from dataset import *

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
            data_summarisation = load_dataset("YashaP/Summarisation_dataset")
            data = data_summarisation["train"]["input"]
            final_data = self.process_data_selection(data)
        
        return final_data
    
    def process_data_selection(self, data):
        
        '''
        The purpose of this function is to select 500 samples from each dataset of their average length
        - TinyStories: 177
        - Alpaca: 50
        - Summarisation: 1100
        '''
        
        if self.data_name == "tinystories":
            shuffled_text = random.sample(data['validation']['text'], len(data['validation']['text']))
            new_data = []
            for sent in shuffled_text:
                if len(sent) == 177:
                    new_data.append(sent)
                if len(new_data) == 500:
                    break
                
        elif self.data_name == "alpaca":
            shuffled_text = random.sample(data["train"]["instruction"])
            new_data = []
            for sent in shuffled_text:
                if len(sent) == 50:
                    new_data.append(sent)
                if len(new_data) == 500:
                    break
        elif self.data_name == "summarisation":
            shuffled_text = random.sample(data['train']['input'])
            new_data = []
            for sent in shuffled_text:
                if len(sent) == 1100:
                    new_data.append(sent)
                if len(new_data) == 500:
                    break
        
        return new_data

if __name__ == "__main__":
    
    #TODO: 
    '''
    1. Write the code to setup Pythia. 
    2. Setup the code to load all the datasets. 
    3. Write the code to get the activations and save it in the figure. 
    4. Write the code to get the activations and save it in the figure. 
    5. Write the code to run the code for all the selected models. 
    '''
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type = str, default="Pythia70m")
    parser.add_argument("--batch_size", type=int, default="16")
    parser.add_argument("--data", type = str, default = "tinystories")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    batch_size = args.batch_size
    dataset_used = args.data
    
    configuration = config(device = "cuda", model_name = model_name, data_name = dataset_used)
    data = configuration.data_selection()
    model = configuration.model_selection()
    
    
    
    
    
    
    