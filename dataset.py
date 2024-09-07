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

    return train_dataset, validation_dataset

def process_data(
    model,
    train_data,
    val_data,
):
    tokenized_train_data = model.tokenizer(train_data['text'], padding=True, truncation=True, return_tensors="pt")
    tokenized_val_data = model.tokenizer(val_data['text'], padding=True, truncation=True, return_tensors="pt")

    train_dataloader = DataLoader(tokenized_train_data, batch_size=16, shuffle=True)
    validation_dataloader = DataLoader(tokenized_val_data, batch_size=16)

    return train_dataloader, validation_dataloader
