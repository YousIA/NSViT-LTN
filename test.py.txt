import torch
from utils import load_dataset
from models import VisionTransformer
from sklearn.metrics import accuracy_score

def evaluate(model, test_loader):
    """
    Evaluate the Vision Transformer model on a test dataset.

    Args:
    - model (nn.Module): Trained Vision Transformer model.
    - test_loader (DataLoader): DataLoader object for the test dataset.

    Returns:
    - accuracy (float): Accuracy of the model on the test dataset.

    """
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    true_labels = []
    predicted_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            logits = model(images)
            _, predicted = torch.max(logits, 1)
            
            # Collect true and predicted labels
            true_labels.extend(labels.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
    print(f'Test Accuracy: {accuracy:.4f}')

    return accuracy

if __name__ == '__main__':
    # Example usage
    test_loader = load_dataset(train=False)  # Load test dataset
    model = VisionTransformer(num_classes=10)  # Initialize model
    checkpoint = torch.load('checkpoint.pth.tar')  # Load trained model checkpoint
    model.load_state_dict(checkpoint['state_dict'])  # Load model weights
    evaluate(model, test_loader)  # Evaluate model on test dataset

