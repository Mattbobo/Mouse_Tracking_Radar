import os
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # Progress bar

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Hyperparameters and Paths
# -------------------------------
TRAINING_DATA_DIR = r"C:\Users\matt\Desktop\degree_trainer\data\processed_data\training_dataset.npz"
VAL_DATA_DIR = r"C:\Users\matt\Desktop\degree_trainer\data\processed_data\val_dataset.npz"
WINDOW_SIZE = 30
STEP_SIZE = 1
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-4
NUM_CLASSES = 4  # Four classes: background, patpat, come, wavewave

# Create a timestamped folder for saving models
timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
MODEL_SAVE_PATH = os.path.join("output", "models", timestamp)
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Define gesture types (order must match training)
gesture_types = ['background', 'patpat', 'come', 'wavewave']


# -------------------------------
# Data Loading Function
# -------------------------------
def load_data(data_path):
    """
    Load the .npz data file from the given path and prepare data X and soft labels y.

    Args:
        data_path (str): Path to the .npz file.

    Returns:
        X (np.array): Array of shape (num_samples, 2, WINDOW_SIZE, 32, 32)
        y (np.array): Array of shape (num_samples, 4) containing soft labels.
    """
    data = np.load(data_path)
    features = data['features']  # Expected shape: (2, 32, 32, frames, ?)

    # Rearrange dimensions and flatten extra dimension into frames.
    features = np.transpose(features, (1, 2, 3, 0, 4))
    features = features.reshape(features.shape[0], features.shape[1], features.shape[2], -1)

    labels = data['labels']  # (frames,)
    labels = labels.reshape(-1)
    ground_truths = data['ground_truths']  # (frames, 4) -> [background, patpat, come, wavewave]
    ground_truths = ground_truths.reshape(-1, ground_truths.shape[2])

    print(f"Loading data from: {data_path}")
    print("features shape:", features.shape)
    print("labels shape:", labels.shape)
    print("ground_truths shape:", ground_truths.shape)

    X_list, y_list = [], []
    num_frames = features.shape[-1]

    # Slide over frames with given window and step size
    for start in range(0, num_frames - WINDOW_SIZE + 1, STEP_SIZE):
        end = start + WINDOW_SIZE
        mid = start + WINDOW_SIZE // 2  # Use middle frame for soft label

        window_feature = features[..., start:end]  # Shape: (2, 32, 32, WINDOW_SIZE)
        label_soft = ground_truths[mid]  # Shape: (4,)

        # Rearrange dimensions to (2, WINDOW_SIZE, 32, 32)
        window_feature = np.transpose(window_feature, (0, 3, 1, 2))
        X_list.append(window_feature)
        y_list.append(label_soft)

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)

    print("X shape:", X.shape)  # (num_samples, 2, WINDOW_SIZE, 32, 32)
    print("y shape:", y.shape)  # (num_samples, 4)
    return X, y


# -------------------------------
# 3D CNN Model Definition
# -------------------------------
class Gesture3DCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(Gesture3DCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(in_channels=2, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(32),

            nn.Conv3d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(64),

            nn.Conv3d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.BatchNorm3d(128),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return torch.softmax(x, dim=1)  # Output probability distribution


# -------------------------------
# Model Training Function
# -------------------------------
def train_model(X_train, y_train, X_val, y_val):
    """
    Train the 3D CNN model using the provided training and validation data.

    Args:
        X_train (np.array): Training features.
        y_train (np.array): Training soft labels.
        X_val (np.array): Validation features.
        y_val (np.array): Validation soft labels.

    Returns:
        model: The trained model.
        history: A dictionary with training/validation losses and accuracies.
    """
    X_train_tensor = torch.from_numpy(X_train).float()
    print(X_train_tensor.shape)  # (7737, 2, 30, 32, 32)
    y_train_tensor = torch.from_numpy(y_train).float()
    X_val_tensor = torch.from_numpy(X_val).float()
    y_val_tensor = torch.from_numpy(y_val).float()

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True,
                              persistent_workers=True)

    model = Gesture3DCNN(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        epoch_train_correct = 0
        epoch_train_total = 0

        train_pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{EPOCHS}] (Training)", unit="batch")
        for batch_x, batch_y in train_pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)

            # Calculate batch accuracy by converting soft labels to hard labels
            preds = torch.argmax(outputs, dim=1)
            targets = torch.argmax(batch_y, dim=1)
            epoch_train_correct += (preds == targets).sum().item()
            epoch_train_total += batch_x.size(0)

            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        train_acc = epoch_train_correct / epoch_train_total
        train_accuracies.append(train_acc)

        # Validation
        model.eval()
        val_running_loss = 0.0
        epoch_val_correct = 0
        epoch_val_total = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(device), val_y.to(device)
                val_outputs = model(val_x)
                val_loss = criterion(val_outputs, val_y)
                val_running_loss += val_loss.item() * val_x.size(0)

                # Compute validation accuracy
                val_preds = torch.argmax(val_outputs, dim=1)
                val_targets = torch.argmax(val_y, dim=1)
                epoch_val_correct += (val_preds == val_targets).sum().item()
                epoch_val_total += val_x.size(0)

        val_epoch_loss = val_running_loss / len(val_dataset)
        val_losses.append(val_epoch_loss)
        val_acc = epoch_val_correct / epoch_val_total
        val_accuracies.append(val_acc)

        print(f"[Epoch {epoch + 1}/{EPOCHS}] Train Loss: {epoch_loss:.4f} || Val Loss: {val_epoch_loss:.4f} || "
              f"Train Acc: {train_acc:.4f} || Val Acc: {val_acc:.4f}")

        # Save the best model based on validation loss
        if val_epoch_loss < best_val_loss:
            best_val_loss = val_epoch_loss
            best_model_path = os.path.join(MODEL_SAVE_PATH, f"epoch_{epoch + 1}_valLoss_{val_epoch_loss:.4f}.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")

    final_model_path = os.path.join(MODEL_SAVE_PATH, "3d_cnn_model.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"Training complete. Final model saved at {final_model_path}")

    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accuracies,
        'val_acc': val_accuracies
    }
    return model, history


# -------------------------------
# Plot Training History
# -------------------------------
def plot_history(history):
    """
    Plot training and validation loss and accuracy curves.
    """
    epochs_range = range(1, len(history['train_loss']) + 1)
    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curves
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['train_acc'], label='Training Accuracy')
    plt.plot(epochs_range, history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()


# -------------------------------
# Main Function
# -------------------------------
def main():
    # Load training and validation data from separate files
    X_train, y_train = load_data(TRAINING_DATA_DIR)
    X_val, y_val = load_data(VAL_DATA_DIR)
    model, history = train_model(X_train, y_train, X_val, y_val)
    plot_history(history)


if __name__ == "__main__":
    main()
