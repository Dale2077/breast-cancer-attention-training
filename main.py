import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

# ===========================
# Configuration Class
# ===========================
class Config:
    """Configuration parameters for the training process."""
    data_dir = os.path.join(os.path.dirname(__file__), "datasets")  # Dataset directory
    batch_size = 32                      # Number of samples per batch
    num_workers = 4                      # Number of subprocesses for data loading
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    epochs = 50                          # Total number of training epochs
    learning_rate = 0.001                # Learning rate for the optimizer
    weight_decay = 1e-4                  # Weight decay (L2 regularization) for the optimizer
    num_classes = 2                      # Number of output classes
    input_size = 224                     # Input image size (height and width)

# ===========================
# Custom Dataset Class
# ===========================
class BreastCancerDataset(Dataset):
    """
    Dataset class for breast cancer images.
    
    Args:
        images (list): List of image file paths.
        labels (list): Corresponding label list.
        transform (callable, optional): Optional image transformations.
    """
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.images)

    def __getitem__(self, idx):
        """Get the image and label at the specified index."""
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Open image and convert to RGB
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            
        return image, label

# ===========================
# Attention Module
# ===========================
class AttentionModule(nn.Module):
    """
    Simple attention module implemented using convolutional layers.
    
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(AttentionModule, self).__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, kernel_size=1),
            nn.BatchNorm2d(in_channels // 8),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply attention weights to the input feature maps."""
        attention_weights = self.attention(x)
        return x * attention_weights

# ===========================
# CNN Model
# ===========================
class CNN(nn.Module):
    """
    Convolutional Neural Network model.
    
    Args:
        num_classes (int): Number of output classes.
        use_attention (bool): Whether to use the attention module.
    """
    def __init__(self, num_classes=2, use_attention=False):
        super(CNN, self).__init__()
        self.use_attention = use_attention
        
        # Define the feature extraction part of CNN
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Conv layer 1
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # Max pooling
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# Conv layer 2
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # Max pooling
            nn.Conv2d(128, 256, kernel_size=3, padding=1),# Conv layer 3
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # Max pooling
        )
        
        # Optional attention module
        if use_attention:
            self.attention = AttentionModule(256)
            
        # Define the classifier part of CNN
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),                # Adaptive average pooling
            nn.Flatten(),                                # Flatten tensor
            nn.Linear(256, num_classes)                  # Fully connected layer
        )

    def forward(self, x):
        """Define the forward propagation of CNN."""
        x = self.features(x)
        if self.use_attention:
            x = self.attention(x)
        x = self.classifier(x)
        return x

# ===========================
# ResNet Basic Block
# ===========================
class ResBlock(nn.Module):
    """
    Basic residual block for ResNet.
    
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int): Stride of the first convolutional layer.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # If dimensions change, match dimensions via convolution
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        """Define the forward propagation of the residual block."""
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add shortcut connection
        out = torch.relu(out)
        return out

# ===========================
# ResNet Model
# ===========================
class ResNet(nn.Module):
    """
    ResNet model.
    
    Args:
        num_classes (int): Number of output classes.
        use_attention (bool): Whether to use the attention module.
    """
    def __init__(self, num_classes=2, use_attention=False):
        super(ResNet, self).__init__()
        self.use_attention = use_attention
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self.make_layer(64, 64, 2)      # Layer 1
        self.layer2 = self.make_layer(64, 128, 2, stride=2)  # Layer 2
        self.layer3 = self.make_layer(128, 256, 2, stride=2) # Layer 3
        
        # Optional attention module
        if use_attention:
            self.attention = AttentionModule(256)
            
        # Global average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        Create a layer consisting of multiple residual blocks.
        
        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            blocks (int): Number of residual blocks.
            stride (int): Stride of the first residual block.
        
        Returns:
            nn.Sequential: A sequential container composed of multiple residual blocks.
        """
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        """Define the forward propagation of ResNet."""
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        if self.use_attention:
            x = self.attention(x)
            
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ===========================
# Training Function
# ===========================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    """
    Train and validate the model.
    
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer.
        num_epochs (int): Number of training epochs.
        device (torch.device): Training device.
    
    Returns:
        tuple: Lists of training losses, training accuracies, validation losses, 
               validation accuracies, and a dictionary of best validation metrics.
    """
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Track best validation accuracy
    best_val_acc = 0.0
    best_metrics = {}

    for epoch in range(num_epochs):
        # ===========================
        # Training Phase
        # ===========================
        model.train()  # Set model to training mode
        train_loss = 0.0
        correct = 0
        total = 0
        
        # Training progress bar
        train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()            # Clear gradients
            outputs = model(images)          # Forward propagation
            loss = criterion(outputs, labels)# Compute loss
            
            loss.backward()                  # Backward propagation
            optimizer.step()                 # Update parameters
            
            train_loss += loss.item()        # Accumulate loss
            _, predicted = outputs.max(1)    # Get predicted class
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Update progress bar with loss and accuracy
            train_bar.set_postfix({
                'Loss': f'{train_loss/len(train_loader):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        # Calculate average training loss and accuracy for this epoch
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(100. * correct / total)
        
        # ===========================
        # Validation Phase
        # ===========================
        model.eval()  # Set model to evaluation mode
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():  # Disable gradient computation
            val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels in val_bar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)          # Forward propagation
                loss = criterion(outputs, labels)# Compute loss
                
                val_loss += loss.item()          # Accumulate loss
                _, predicted = outputs.max(1)    # Get predicted class
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # Update progress bar with loss and accuracy
                val_bar.set_postfix({
                    'Loss': f'{val_loss/len(val_loader):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        # Calculate average validation loss and accuracy for this epoch
        val_losses.append(val_loss / len(val_loader))
        val_accs.append(100. * correct / total)
        
        # ===========================
        # Record Best Validation Metrics
        # ===========================
        # If current validation accuracy is the best, record it
        if val_accs[-1] > best_val_acc:
            best_val_acc = val_accs[-1]
            best_metrics = {
                'Epoch': epoch + 1,
                'Train Loss': train_losses[-1],
                'Train Acc': train_accs[-1],
                'Val Loss': val_losses[-1],
                'Val Acc': val_accs[-1]
            }
        
        # ===========================
        # Print Epoch Summary
        # ===========================
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accs[-1]:.2f}%')
        print(f'Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accs[-1]:.2f}%')
        print('-' * 50)
    
    return train_losses, train_accs, val_losses, val_accs, best_metrics

# ===========================
# Plotting Function
# ===========================
def plot_training(train_losses, train_accs, val_losses, val_accs, model_name):
    """
    Plot training and validation loss and accuracy curves.
    
    Args:
        train_losses (list): List of training losses per epoch.
        train_accs (list): List of training accuracies per epoch.
        val_losses (list): List of validation losses per epoch.
        val_accs (list): List of validation accuracies per epoch.
        model_name (str): Model name (used for chart title and saved filename).
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_training.png')  # Save as image file
    plt.close()

# ===========================
# Main Function
# ===========================
def main():
    """Main function to execute the training pipeline."""
    config = Config()
    
    # ===========================
    # Data Preprocessing
    # ===========================
    transform = transforms.Compose([
        transforms.Resize((config.input_size, config.input_size)),  # Resize image
        transforms.ToTensor(),                                     # Convert to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406],          # Normalize using ImageNet mean
                             std=[0.229, 0.224, 0.225])           # Normalize using ImageNet std
    ])
    
    # ===========================
    # Data Loading
    # ===========================
    images = []
    labels = []
    for class_idx, class_name in enumerate(['benign', 'malignant']):
        class_dir = os.path.join(config.data_dir, class_name)  # Class directory path
        for img_name in os.listdir(class_dir):
            # Check for valid image file extensions
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
                images.append(os.path.join(class_dir, img_name))  # Add image path
                labels.append(class_idx)                          # Add corresponding label
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = BreastCancerDataset(X_train, y_train, transform)
    val_dataset = BreastCancerDataset(X_val, y_val, transform)
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=config.num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=config.num_workers
    )
    
    # ===========================
    # Model Initialization
    # ===========================
    models = {
        'CNN': CNN(config.num_classes, use_attention=False),
        'CNN_Attention': CNN(config.num_classes, use_attention=True),
        'ResNet': ResNet(config.num_classes, use_attention=False),
        'ResNet_Attention': ResNet(config.num_classes, use_attention=True)
    }
    
    # Store best metrics for each model
    all_best_metrics = []
    
    # ===========================
    # Train Each Model
    # ===========================
    for model_name, model in models.items():
        print(f'\nTraining model: {model_name}...')
        model = model.to(config.device)                  
        criterion = nn.CrossEntropyLoss()                # Define loss function
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay
        )                                                 # Define optimizer
        
        # Train the model
        train_losses, train_accs, val_losses, val_accs, best_metrics = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            config.epochs, config.device
        )
        
        # Save model state dictionary
        torch.save(model.state_dict(), f'{model_name}.pth')
        
        plot_training(train_losses, train_accs, val_losses, val_accs, model_name)
        
        best_metrics['Model'] = model_name
        all_best_metrics.append(best_metrics)
    
    # ===========================
    # Save Best Metrics to CSV
    # ===========================
    csv_file = 'best_metrics.csv'
    csv_columns = ['Model', 'Epoch', 'Train Loss', 'Train Acc', 'Val Loss', 'Val Acc']
    
    try:
        with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader() 
            for data in all_best_metrics:
                writer.writerow(data)  
        print(f'\nBest metrics for all models saved to {csv_file}')
    except IOError:
        print("I/O error occurred while writing to CSV file")


if __name__ == '__main__':
    main()
