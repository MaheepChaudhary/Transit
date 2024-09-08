from imports import *


def inspect_data(data):

    # data contains "train" and "validation" data as folders and is of data type "dictionary"
    train_dataset = data['train']
    validation_dataset = data['validation']

    # Inspect the first few rows
    pprint(train_dataset[:30])
    pprint(validation_dataset[:30])

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
    # Extract input_ids and attention_mask from each item in the batch
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]

    # Convert lists of tensors to a single tensor
    input_ids = torch.stack(input_ids)
    attention_masks = torch.stack(attention_masks)

    return {'input_ids': input_ids, 'attention_mask': attention_masks}



def tokenize_with_progress(data, tokenizer, max_len=128):
    tokenized_data = {
        'input_ids': [],
        'attention_mask': []
    }

    for text in tqdm(data['text'], desc="Tokenizing", unit="sample"):
        tokenized_sample = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        )
        tokenized_data['input_ids'].append(tokenized_sample['input_ids'].squeeze(0))
        tokenized_data['attention_mask'].append(tokenized_sample['attention_mask'].squeeze(0))

    tokenized_data['input_ids'] = torch.stack(tokenized_data['input_ids'])
    tokenized_data['attention_mask'] = torch.stack(tokenized_data['attention_mask'])

    return tokenized_data

class TokenizedDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data['input_ids']
        self.attention_mask = tokenized_data['attention_mask']

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx]
        }

def process_data(model, train_data, val_data, batch_size):
    max_len = 128  # You can set this to a value that suits your model or dataset.

    # tokenized_train_data = tokenize_with_progress(train_data, model.tokenizer, max_len)
    tokenized_val_data = tokenize_with_progress(val_data, model.tokenizer, max_len)

    # train_dataset = TokenizedDataset(tokenized_train_data)
    val_dataset = TokenizedDataset(tokenized_val_data)

    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=custom_collate_fn)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=custom_collate_fn, pin_memory=True, num_workers=1)
    # return train_dataloader, validation_dataloader
    return validation_dataloader

