import argparse
from train import train
from test import evaluate
from utils import load_dataset
from models import VisionTransformer
import torch.optim as optim

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manufacturing Defect Detection')
    parser.add_argument('--mode', type=str, default='train', help='train or test mode')
    args = parser.parse_args()

    if args.mode == 'train':
        train_loader = load_dataset()
        model = VisionTransformer(num_classes=12)  # Adjust based on your number of classes
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        num_epochs = 10
        lambda_ltn = 0.1
        train(model, train_loader, optimizer, num_epochs, lambda_ltn)
    elif args.mode == 'test':
        test_loader = load_dataset()  # Adjust to load your test dataset
        model = VisionTransformer(num_classes=12)  # Adjust based on your number of classes
        evaluate(model, test_loader)
    else:
        print('Invalid mode. Please choose either "train" or "test".')
