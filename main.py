from imports import *
from models import *
from dataset import *
from norm_and_grad import *

def create_dataloader(tokenized_data, batch_size=2):
    input_ids = torch.tensor([item['input_ids'] for item in tokenized_data])
    attention_mask = torch.tensor([item['attention_mask'] for item in tokenized_data])

    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader

# Create DataLoader


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", type=int, default="8")
    parser.add_argument("--title", type=str, default="Title of the graph")
    parser.add_argument("--model", type=str, default="pythia")
    
    args = parser.parse_args()
    name = args.title
    
    if args.model == "pythia":
        model = LanguageModel("EleutherAI/pythia-70m", device_map="cpu")
    elif args.model == "gpt2":
        model = LanguageModel("openai-community/gpt2", device_map = "cpu")
    
    train_data, val_data = inspect_data(data)
    print();print(model);print()
    # Tokenizing the text data
    try:
        with open(f"data/val_data_b{args.batch_size}.pkl", "rb") as f:
            val_dataloader = pickle.load(f)

    except:
        val_dataloader = process_data(model, train_data, val_data, args.batch_size)
        
        with open(f"data/val_data_b{args.batch_size}.pkl", "wb") as f:
            pickle.dump(val_dataloader, f)
    
    
    
    if args.model == "pythia":
        
        if args.title == "post_mlp_addn_resid_layer":
            normed_class = act_pythia_resid_post_mlp_addn(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            normed_class.norm()
            
            grad_class = grad_pythia_resid_post_mlp_addn(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            grad_class.grad_norm()
        
        elif args.title == "mlp_output":
            
            normed_class_mlp = act_pythia_mlp(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            normed_class_mlp.norm()
        
        # grad_class_mlp = grad_pythia_mlp(
        #     title=args.title, 
        #     name = name, 
        #     model = model, 
        #     dataloader = val_dataloader, 
        #     )
        
        # grad_class_mlp.grad_norm()
    
    elif args.model == "gpt2":
        pass
        # normed_class = normed_gpt2(
        #     title=args.title, 
        #     name = name, 
        #     model = model, 
        #     dataloader = val_dataloader, 
        #     batch_size=args.batch_size)
        
        # normed_class.norm()
        
        # grad_class = grad_gpt2(
        #     title=args.title, 
        #     name = name, 
        #     model = model, 
        #     dataloader = val_dataloader, 
        #     batch_size=args.batch_size)
        
        # grad_class.grad_norm()
    

    
    # normed_grad = gradients_norm(model, val_dataloader, args.title, name)
    # normed_grad.norm()
    # normed_grad.normwmean()
    
    # normed_single = single_sample_act_norm(model, val_data)
    # normed_single.norm()
    
    # img_concat()
    
    # grad_normed_single = single_sample_grad_norm(model, train_data)
    # grad_normed_single.grad_norm()
    
    # img_concat()