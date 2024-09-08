from imports import *


def inspect_data():

    # data contains "train" and "validation" data as folders and is of data type "dictionary"
    train_dataset = data['train']
    validation_dataset = data['validation']

    # Inspect the first few rows
    pprint(train_dataset[:5])
    pprint(validation_dataset[:5])

    # Get the total number of rows
    pprint(f"Training set size: {len(train_dataset)}")
    pprint(f"Validation set size: {len(validation_dataset)}")

    # Get the column names and types
    pprint(train_dataset.column_names)
    pprint(train_dataset.features)
    print()
    pprint("---------------------------------")
    print()
    return train_dataset, validation_dataset

def custom_collate_fn(batch):
    # Implement conversion from tokenizers.Encoding to PyTorch tensors
    # Example:
    input_ids = [item.ids for item in batch]
    attention_masks = [item.attention_mask for item in batch]
    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
    return {'input_ids': input_ids, 'attention_mask': attention_masks}



def process_data(
    model,
    train_data,
    val_data,
):
    
    max_len = 128  # You can set this to a value that suits your model or dataset.
    tokenized_train_data = model.tokenizer(train_data['text'], padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
    tokenized_val_data = model.tokenizer(val_data['text'], padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")

    train_dataloader = DataLoader(tokenized_train_data, batch_size=8, shuffle=True, collate_fn = custom_collate_fn)
    validation_dataloader = DataLoader(tokenized_val_data, batch_size=8, collate_fn=custom_collate_fn)

    return train_dataloader, validation_dataloader
