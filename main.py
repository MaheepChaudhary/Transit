from imports import *
from models import *
from dataset import *

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
    
    args = parser.parse_args()
    
    
    # model = LanguageModel("EleutherAI/pythia-70m", device_map=t.device("cuda" if t.cuda.is_available() else "mps"))
    model = LanguageModel("EleutherAI/pythia-70m", device_map="cpu")
    # model = LanguageModel("openai-community/gpt2", device_map = "cpu")
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
    
    # Computing the activations
    try:
        with open("data/activation_embeds_post_mlp_addn_resid.pkl", "rb") as f:
            activation_embeds = pickle.load(f)
        with open("mdata/activation_embeds_post_mlp_addn_resid.pkl", "rb") as f_:
            activation_embeds_mean = pickle.load(f_)
    except:
        activation_embeds = activation_embeds_fn(model, val_dataloader, args.batch_size)
        with open("mdata/activation_embeds_post_mlp_addn_resid.pkl", "wb") as f:
            pickle.dump(activation_embeds, f)
    
    name = args.title
    
    normed_class = normed(args.title, name)
    normed_class.normwmean(activation_embeds_mean)
    normed_class.norm(activation_embeds)
    
    # normed_grad = gradients_norm(model, val_dataloader, args.title, name)
    # normed_grad.norm()
    # normed_grad.normwmean()
    
    # normed_single = single_sample_act_norm(model, val_data)
    # normed_single.norm()
    
    # img_concat()
    
    # grad_normed_single = single_sample_grad_norm(model, train_data)
    # grad_normed_single.grad_norm()
    
    # img_concat()