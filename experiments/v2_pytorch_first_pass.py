import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision.transforms as transforms
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Configuration
IMG_HEIGHT, IMG_WIDTH = 48, 48
BATCH_SIZE = 128
EPOCHS = 50
NUM_CLASSES = 7
LEARNING_RATE = 0.005
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure dataset class order matches folder names
EXPRESSIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

print(f"Using device: {DEVICE}")

# ==================== DATASET CLASS ====================
class FERDataset(Dataset):
    """Custom Dataset for FER-2013 images (no internal transform by default)"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # list of (path, label)

        for label_idx, emotion in enumerate(EXPRESSIONS):
            emotion_dir = os.path.join(root_dir, emotion)
            if not os.path.isdir(emotion_dir):
                continue
            for fname in sorted(os.listdir(emotion_dir)):
                if fname.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.samples.append((os.path.join(emotion_dir, fname), label_idx))

        print(f"Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('L')  # grayscale
        if self.transform:
            img = self.transform(img)
        return img, label

class SubsetWithTransform(Dataset):
    """Wrap a Subset and apply a transform different from underlying dataset."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]  # img returned without transform if underlying had None
        if self.transform:
            img = self.transform(img)
        return img, label

# ==================== MODEL ====================
class EmotionCNN(nn.Module):
    """CNN model for emotion classification"""

    def __init__(self, num_classes=NUM_CLASSES):
        super(EmotionCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Block 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Block 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Block 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )

        # Fully connected layers (expects 48 -> 24 -> 12 -> 6 -> 3 spatial dims)
        self.fc = nn.Sequential(
            nn.Linear(512 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# ==================== DATA LOADING ====================
def create_data_loaders(data_dir, batch_size=BATCH_SIZE, val_split=0.2, num_workers=0):
    """Create train, validation, and test data loaders"""
    train_root = os.path.join(data_dir, 'train')
    test_root = os.path.join(data_dir, 'test')

    if not os.path.isdir(train_root) or not os.path.isdir(test_root):
        raise FileNotFoundError(f"train/ or test/ not found inside {data_dir}")

    # transforms
    base_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.15, 0.15)),
        transforms.RandomResizedCrop(IMG_HEIGHT, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # create base dataset without transforms (so we can apply different transforms to subsets)
    base_train = FERDataset(train_root, transform=None)
    n_total = len(base_train)
    if n_total == 0:
        raise RuntimeError(f"No training images found in {train_root}")

    n_val = int(n_total * val_split)
    n_train = n_total - n_val
    generator = torch.Generator().manual_seed(123)
    train_subset, val_subset = random_split(base_train, [n_train, n_val], generator=generator)

    # wrap subsets with respective transforms
    train_ds = SubsetWithTransform(train_subset, transform=train_transform)
    val_ds = SubsetWithTransform(val_subset, transform=base_transform)

    test_ds = FERDataset(test_root, transform=base_transform)

    pin_memory = True if DEVICE.type == 'cuda' else False

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)

    print(f"\n✓ Data loaders created:")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Test samples: {len(test_ds)}")
    print(f"  Classes: {EXPRESSIONS}")

    return train_loader, val_loader, test_loader

# ==================== TRAINING ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({'loss': f"{running_loss/total:.4f}", 'acc': f"{100.*correct/total:.2f}%"})

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total if total else 0.0
    epoch_acc = 100. * correct / total if total else 0.0
    return epoch_loss, epoch_acc

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCHS):
    """Complete training loop"""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    patience = 15
    patience_counter = 0

    print("\n" + "="*60)
    print("Starting Training...")
    print("="*60)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_fer2013_model.pth')
            print(f"✓ Best model saved! Val Acc: {val_acc:.2f}%")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break

    print("\n" + "="*60)
    print(f"Training Complete! Best Val Acc: {best_val_acc:.2f}%")
    print("="*60)
    return history

# ==================== EVALUATION ====================
def evaluate_model(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    print("\n" + "="*60)
    print("Evaluating on Test Set...")
    print("="*60)

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing', leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    test_acc = 100. * correct / total if total else 0.0
    print(f"\n✓ Test Accuracy: {test_acc:.2f}%")

    print("\n" + "="*60)
    print("Classification Report:")
    print("="*60)
    print(classification_report(all_labels, all_preds, target_names=[e.capitalize() for e in EXPRESSIONS]))

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[e.capitalize() for e in EXPRESSIONS],
                yticklabels=[e.capitalize() for e in EXPRESSIONS])
    plt.title('Confusion Matrix - FER2013 Test Set', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()

    return test_acc, cm

def plot_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    print("✓ Training history saved as 'training_history.png'")
    plt.show()

def predict_single_image(model, image_path, device=DEVICE):
    """Predict emotion from a single image"""
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    image = Image.open(image_path).convert('L')
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities[0][predicted_class].item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Input Image', fontsize=12, fontweight='bold')
    ax1.axis('off')

    ax2.barh([e.capitalize() for e in EXPRESSIONS], probabilities[0].cpu().numpy())
    ax2.set_xlabel('Probability')
    ax2.set_title('Emotion Predictions', fontsize=12, fontweight='bold')
    ax2.set_xlim([0, 1])

    plt.tight_layout()
    plt.show()

    return EXPRESSIONS[predicted_class], confidence

# ==================== MAIN ====================
if __name__ == "__main__":

    print("\n" + "="*60)
    print("FER-2013 FACIAL EXPRESSION RECOGNITION - PyTorch")
    print("="*60)

    # Auto-detect dataset folder that contains 'train/' and 'test/'
    script_dir = os.path.dirname(__file__)
    candidates = [
        script_dir,
        os.path.join(script_dir, 'fer2013'),
        os.path.join(script_dir, 'data'),
        os.path.join(script_dir, 'dataset'),
        os.path.join(script_dir, '..'),
    ]
    DATA_DIR = None
    for c in candidates:
        if os.path.isdir(os.path.join(c, 'train')) and os.path.isdir(os.path.join(c, 'test')):
            DATA_DIR = os.path.abspath(c)
            break

    if DATA_DIR is None:
        print("Error: could not find dataset folder containing 'train/' and 'test/'.")
        print("Checked locations:")
        for c in candidates:
            print("  -", os.path.abspath(c))
        print("\nPlace your dataset so it contains 'train/' and 'test/' subfolders,")
        print("or set DATA_DIR manually in the script to the folder that contains them.")
        sys.exit(1)

    print(f"\nUsing DATA_DIR = {DATA_DIR}")

    # Step 1: Create data loaders
    print("\n" + "="*60)
    print("Loading Data...")
    print("="*60)
    try:
        # Use num_workers=0 for Windows compatibility; change if you know your environment supports more.
        train_loader, val_loader, test_loader = create_data_loaders(DATA_DIR, batch_size=BATCH_SIZE, num_workers=0)
    except Exception as e:
        print(f"Error preparing data loaders: {e}")
        sys.exit(1)

    # Step 2: Create model
    print("\n" + "="*60)
    print("Building Model...")
    print("="*60)
    model = EmotionCNN(num_classes=NUM_CLASSES).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Step 3: Define loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Step 4: Train model
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=EPOCHS)

    # Step 5: Plot training history
    print("\n" + "="*60)
    print("Plotting Training History...")
    print("="*60)
    plot_history(history)

    # Step 6: Load best model and evaluate
    print("\n" + "="*60)
    print("Loading Best Model...")
    print("="*60)
    if os.path.exists('best_fer2013_model.pth'):
        model.load_state_dict(torch.load('best_fer2013_model.pth', map_location=DEVICE))
    else:
        print("Best model not found, evaluating current model.")

    test_acc, cm = evaluate_model(model, test_loader, DEVICE)

    # Step 7: Save final model
    print("\n" + "="*60)
    print("Saving Final Model...")
    print("="*60)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'history': history
    }, 'fer2013_final_model.pth')
    print("✓ Model saved as 'fer2013_final_model.pth'")

    print("\n" + "="*60)
    print("TRAINING COMPLETE! 🎉")
    print("="*60)
    print(f"Final Test Accuracy: {test_acc:.2f}%")
    print("\nFiles created:")
    print("  📁 best_fer2013_model.pth (best model weights)")
    print("  📁 fer2013_final_model.pth (final checkpoint)")
    print("  📊 training_history.png")
    print("  📊 confusion_matrix.png")
    
    # Example usage
    print("\n" + "="*60)
    print("HOW TO USE THE TRAINED MODEL:")
    print("="*60)
    print("""
# Load the model
import torch
from your_script import EmotionCNN

model = EmotionCNN(num_classes=7)
model.load_state_dict(torch.load('best_fer2013_model.pth'))
model.eval()

# Predict on a single image
emotion, confidence = predict_single_image(model, 'your_image.jpg', device)
print(f"Predicted: {emotion} ({confidence*100:.1f}%)")
    """)