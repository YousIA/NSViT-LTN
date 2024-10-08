import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

def load_dataset(train=True, batch_size=16, shuffle=True):
    """
    Load and preprocess the dataset.

    Args:
    - train (bool): Whether to load the training dataset (default: True).
    - batch_size (int): Batch size for DataLoader (default: 16).
    - shuffle (bool): Whether to shuffle the data (default: True).

    Returns:
    - train_loader (DataLoader): DataLoader object for the loaded dataset.

    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])


    if train:
        dataset = datasets.Data(transform=transform)
    else:
        dataset = datasets.Data(transform=transform, train=False)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return train_loader

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save model checkpoint.

    Args:
    - state (dict): Model state dictionary to be saved.
    - filename (str): File name for saving checkpoint (default: 'checkpoint.pth.tar').

    """
    torch.save(state, filename)

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    """
    Load model checkpoint.

    Args:
    - model (nn.Module): Model to which the checkpoint parameters will be loaded.
    - optimizer (torch.optim.Optimizer): Optimizer to which the checkpoint state will be loaded.
    - filename (str): File name of the saved checkpoint (default: 'checkpoint.pth.tar').

    Returns:
    - start_epoch (int): Starting epoch from which training should resume.

    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    return start_epoch


def load_rules_from_json(json_file):
    """
    Load logical rules from a JSON file.

    Args:
    - json_file (str): Path to the JSON file containing rules.

    Returns:
    - rules (list): List of rules loaded from the JSON file.

    """
    with open(json_file, 'r') as f:
        rules_data = json.load(f)
        rules = rules_data.get('rules', [])
    return rules
