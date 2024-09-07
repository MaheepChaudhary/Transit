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

def process_data(
    model,
    train_data,
    val_data,
):
    max_len = 128  # You can set this to a value that suits your model or dataset.

    # Tokenizing validation data with max_length specified
    tokenized_val_data = [
        model.tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
        for text in tqdm(val_data['text'], desc="Tokenizing Validation Data")
    ]

    # Tokenizing training data with max_length specified
    # tokenized_train_data = [
    #     model.tokenizer(text, padding='max_length', truncation=True, max_length=max_len, return_tensors="pt")
    #     for text in tqdm(train_data['text'], desc="Tokenizing Training Data")
    # ]

    # Convert the list of tokenized data back to a suitable format
    # tokenized_train_data = {key: torch.cat([item[key] for item in tokenized_train_data]) for key in tokenized_train_data[0]}
    tokenized_val_data = {key: torch.cat([item[key] for item in tokenized_val_data]) for key in tokenized_val_data[0]}

    # train_dataloader = DataLoader(tokenized_train_data, batch_size=8, shuffle=True)
    validation_dataloader = DataLoader(tokenized_val_data, batch_size=8)

    # return train_dataloader, validation_dataloader
    return validation_dataloader
