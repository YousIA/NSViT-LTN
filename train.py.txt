import torch
import torch.optim as optim
from utils import load_dataset, load_rules_from_json
from models import VisionTransformer
import json

def train(model, train_loader, optimizer, num_epochs, lambda_ltn, rules):
    model.train()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images)
            predictions = {
                'IsDefectiveScrew': logits[:, 0],
                'HasRust': logits[:, 1],
                'IsDeformed': logits[:, 2],
                'IsDefectiveWood': logits[:, 3],
                'HasCrack': logits[:, 4],
                'HasKnot': logits[:, 5],
                'IsDefectiveCylinder': logits[:, 6],
                'IsDented': logits[:, 7],
                'HasScratch': logits[:, 8],
                'IsDefectiveCable': logits[:, 9],
                'IsFrayed': logits[:, 10],
                'HasExposedWires': logits[:, 11]
            }
            loss = total_loss_function(logits, labels, predictions, rules, lambda_ltn)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}')

if __name__ == '__main__':
    train_loader = load_dataset()
    model = VisionTransformer(num_classes=12)  # Adjust based on your number of classes
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 10
    lambda_ltn = 0.1
    rules = load_rules_from_json('rules.json')  # Load rules from JSON file
    train(model, train_loader, optimizer, num_epochs, lambda_ltn, rules)
