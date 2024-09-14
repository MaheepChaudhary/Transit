from imports import *
from models import *
from dataset import *
from norm_and_grad import *
from grad_using_hooks import *

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
    elif args.model == "torch_pythia":
        model_name = 'EleutherAI/pythia-70m'
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name).to('cpu')
        model.eval()
        train_data, val_data = inspect_data(data)
        print();print(model);print()
        with open(f"data/pythia_val_data_b{args.batch_size}.pkl", "rb") as f:
            val_dataloader = pickle.load(f)
    
    if args.model == "pythia" or args.model == "gpt2":
        train_data, val_data = inspect_data(data)
        print();print(model);print()
        # Tokenizing the text data
        try:
            with open(f"data/{args.model}_val_data_b{args.batch_size}.pkl", "rb") as f:
                val_dataloader = pickle.load(f)

        except:
            val_dataloader = process_data(model, train_data, val_data, args.batch_size)
            
            with open(f"data/{args.model}_val_data_b{args.batch_size}.pkl", "wb") as f:
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
            normed_class.act_normwmean_ppma()
            
            grad_class = grad_pythia_resid_post_mlp_addn(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            grad_class.grad_norm()
            grad_class.grad_normwmean_ppma()
            
            
        
        elif args.title == "mlp_output":
            
            normed_class_mlp = act_pythia_mlp(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            normed_class_mlp.norm()
            normer_class_mlp = normed_class_mlp.act_normwmean_mlp()
        
            grad_class_mlp = grad_pythia_mlp(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
        
            grad_class_mlp.grad_norm()
            grad_class_mlp.grad_normwmean_mlp()
        
        elif args.title == "attn_output":
            
            normed_class_attention = act_pythia_attention(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            normed_class_attention.norm()
            normed_class_attention.act_normwmean_attn()
        
            grad_class_attention = grad_pythia_attention(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            grad_class_attention.grad_norm()
            grad_class_attention.grad_normwmean_attn()
    
    elif args.model == "gpt2":
        
        if args.title == "post_mlp_addn_resid_layer":
            normed_class = act_gpt2_resid_post_mlp_addn(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            normed_class.norm()
            
            grad_class = grad_gpt2_resid_post_mlp_addn(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            grad_class.grad_norm()
        
        elif args.title == "mlp_output":
            
            normed_class_mlp = act_gpt2_mlp(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            normed_class_mlp.norm()
        
            grad_class_mlp = grad_gpt2_mlp(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
        
            grad_class_mlp.grad_norm()
        
        elif args.title == "attn_output":
            
            normed_class_attention = act_gpt2_attention(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            normed_class_attention.norm()
        
            grad_class_attention = grad_gpt2_attention(
                title=args.title, 
                name = name, 
                model = model, 
                dataloader = val_dataloader, 
                )
            
            grad_class_attention.grad_norm()
    
    elif args.model == "torch_pythia":
        pythia_gradients = pythia_grad(model, tokenizer, dataloader = val_dataloader)
        gradients = pythia_gradients.forward()
        print(np.array(gradients["h[5].attention"]).shape)
        
    
    # normed_grad = gradients_norm(model, val_dataloader, args.title, name)
    # normed_grad.norm()
    # normed_grad.normwmean()
    
    # normed_single = single_sample_act_norm(model, val_data)
    # normed_single.norm()
    
    # img_concat()
    
    # grad_normed_single = single_sample_grad_norm(model, train_data)
    # grad_normed_single.grad_norm()
    
    # img_concat()