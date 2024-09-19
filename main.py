from imports import *
from dataset import *

class config:
    
    def __init__(self, device):
        self.device = device


    def model_selection(self,model_name):
        if model_name == "Pythia14m":
            model = LanguageModel("EleutherAI/pythia-14m", device_map=self.device)
        elif model_name == "Pythia70m":
            model = LanguageModel("EleutherAI/pythia-70m", device_map=self.device)
        elif model_name == "Pythia160m":
            model = LanguageModel("EleutherAI/pythia-160m", device_map=self.device)
        elif model_name == "Pythia410m":
            model = LanguageModel("EleutherAI/pythia-410m", device_map=self.device)
        elif model_name == "Pythia1b":
            model = LanguageModel("EleutherAI/pythia-1b", device_map=self.device)
        elif model_name == "Pythia1.4b":
            model = LanguageModel("EleutherAI/pythia-1.4b", device_map=self.device)
        elif model_name == "Pythia2.8b":
            model = LanguageModel("EleutherAI/pythia-2.8b", device_map=self.device)
        elif model_name == "Pythia6.9b":
            model = LanguageModel("EleutherAI/pythia-6.9b", device_map=self.device)
        elif model_name == "Pythia12b":
            model = LanguageModel("EleutherAI/pythia-12b", device_map=self.device)
        
        return model

    def data_selection(self, data_name):
        
        if data_name == "tinystories":
            pre_data = load_dataset("roneneldan/TinyStories")
            data = pre_data["validation"]['text'] # as we are experimenting on the validation dataset. 
            self.process_data_selection(data, length = 177)
        
        elif data_name == "alpaca":
            pre_data_alpaca = load_dataset("tatsu-lab/alpaca")
            data = pre_data_alpaca["train"]["instruction"]
        
        elif data_name == "summarisation":
            data_summarisation = load_dataset("YashaP/Summarisation_dataset")
            data = data_summarisation["train"]["input"]
    
    def process_data_selection(self, data, length):
        '''
        The purpose of this function is to select 500 samples from each dataset of their average length
        - TinyStories: 177
        - Alpaca: 50
        - Summarisation: 1100
        '''


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
    
    
    
    
    
    
    