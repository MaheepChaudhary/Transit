from imports import *
from dataset import *

def model_selection(model_name, device):
    if model_name == "Pythia14m":
        model = LanguageModel("EleutherAI/pythia-14m", device_map=device)
    elif model_name == "Pythia70m":
        model = LanguageModel("EleutherAI/pythia-70m", device_map=device)
    elif model_name == "Pythia160m":
        model = LanguageModel("EleutherAI/pythia-70m", device_map=device)
    elif model_name == "Pythia410m":
        model = LanguageModel("EleutherAI/pythia-70m", device_map=device)
    elif model_name == "Pythia1b":
        model = LanguageModel("EleutherAI/pythia-70m", device_map=device)
    elif model_name == "Pythia1.4b":
        model = LanguageModel("EleutherAI/pythia-70m", device_map=device)
    elif model_name == "Pythia2.8b":
        model = LanguageModel("EleutherAI/pythia-70m", device_map=device)
    elif model_name == "Pythia6.9b":
        model = LanguageModel("EleutherAI/pythia-70m", device_map=device)
    elif model_name == "Pythia12b":
        model = LanguageModel("EleutherAI/pythia-70m", device_map=device)
    
    return model


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
    
    
    
    
    