import numpy as np
import os
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from dlgo.data.parallel_processor import GoDataProcessor
from dlgo.encoders.simple import SimpleEncoder
from dlgo.network import GoNetwork


class GoDataset(Dataset):
    """PyTorch Dataset wrapper for Go data generator."""
    
    def __init__(self, generator, num_samples, batch_size, num_classes):
        self.generator = generator
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.steps = math.ceil(num_samples / batch_size)
        
    def __len__(self):
        return self.num_samples
    
    def get_generator(self):
        """Returns the underlying generator for DataLoader."""
        return self.generator.generate(self.batch_size, self.num_classes)


def collate_from_generator(batch):
    """Custom collate function that pulls from generator."""
    # batch is ignored, we pull from generator instead
    return batch[0] if batch else None



class GeneratorDataLoader:
    """Wrapper to make generator work with PyTorch training loop."""
    
    def __init__(self, generator, num_samples, batch_size, num_classes):
        self.generator = generator
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.steps = math.ceil(num_samples / batch_size)
        
    def __iter__(self):
        # Use return_tensors=True to get PyTorch tensors directly
        self.gen = self.generator.generate(self.batch_size, self.num_classes, return_tensors=True)
        self.current_step = 0
        return self
    
    def __next__(self):
        if self.current_step >= self.steps:
            raise StopIteration

        x_tensor, y_tensor = next(self.gen)
        
        self.current_step += 1
        return x_tensor, y_tensor
    
    def __len__(self):
        return self.steps


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


def main():
    # Configuration
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    num_games = 1000
    
    batch_size = 128
    epochs = 25
    learning_rate = 0.01
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    encoder = SimpleEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())
    
    train_generator = processor.load_go_data('train', num_games, use_generator=True)
    test_generator = processor.load_go_data('test', num_games, use_generator=True)
    
    train_samples = train_generator.get_num_samples()
    test_samples = test_generator.get_num_samples()
    
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {test_samples}")
    
    # Create data loaders
    train_loader = GeneratorDataLoader(train_generator, train_samples, batch_size, num_classes)
    val_loader = GeneratorDataLoader(test_generator, test_samples, batch_size, num_classes)
    
    # Debug: Check data format
    print("\n=== VERIFY DATA FORMAT ===")
    train_iter = iter(train_loader)
    x_batch, y_batch = next(train_iter)
    print(f"Input shape: {x_batch.shape}")  # Should be (128, 11, 19, 19)
    print(f"Label shape: {y_batch.shape}")  # Should be (128,)
    print(f"Unique labels in batch: {len(torch.unique(y_batch))}")
    print("=== END VERIFY ===\n")
    
    # Create model
    model = GoNetwork(input_channels=encoder.num_planes, num_classes=num_classes)
    model = model.to(device)
    
    # Print model summary
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Setup checkpoints and logging
    checkpoint_dir = Path('../checkpoints')
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    writer = SummaryWriter(log_dir=checkpoint_dir / 'logs')
    
    # Training loop
    best_val_acc = 0.0
    patience_counter = 0
    patience = 10
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('Learning_rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
            }, checkpoint_dir / 'best_model.pth')
            print(f"✓ Saved best model (val_acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save checkpoint every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, checkpoint_dir / f'model_epoch_{epoch:02d}_acc_{val_acc:.4f}.pth')
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs")
            break
    
    # Save final model
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
    }, checkpoint_dir / 'final_model.pth')
    print(f"\nFinal model saved to: {checkpoint_dir / 'final_model.pth'}")
    
    writer.close()
    
    return model, history


def plot_training_history(history, save_path='../checkpoints/training_history.png'):
    """Plot training and validation metrics."""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs_range = range(1, len(history['train_acc']) + 1)
    
    # Plot accuracy
    ax1.plot(epochs_range, history['train_acc'], label='Train')
    ax1.plot(epochs_range, history['val_acc'], label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(epochs_range, history['train_loss'], label='Train')
    ax2.plot(epochs_range, history['val_loss'], label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def load_model(checkpoint_path, device='cpu'):
    """Load a saved model."""
    model = GoNetwork(input_channels=11, num_classes=361)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    model, history = main()
    
    # Plot training history
    try:
        plot_training_history(history)
    except ImportError:
        print("Matplotlib not available for plotting. Install with: pip install matplotlib")
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print("\nTo view training in TensorBoard, run:")
    print("  tensorboard --logdir ../checkpoints/logs")