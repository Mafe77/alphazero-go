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
from dlgo.data.generator import GoDataset, GeneratorDataLoader
from dlgo.encoders.simple import SimpleEncoder
from dlgo.network import GoNetwork

def collate_from_generator(batch):
    """Custom collate function that pulls from generator."""
    # batch is ignored, we pull from generator instead
    return batch[0] if batch else None


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

def debug_samples(processor, data_type, num_samples):
    """Debug what samples will be drawn."""
    from dlgo.data.sampling import Sampler
    from dlgo.data.index_processor import KGSIndex
    
    index = KGSIndex(data_directory=processor.data_dir)
    sampler = Sampler(data_dir=processor.data_dir)
    
    print(f"\n=== Debug {data_type} data ===")
    data = sampler.draw_data(data_type, num_samples)
    print(f"Drew {len(data)} samples for {data_type}")
    
    if data:
        print(f"Sample data entry: {data[0]}")
        
        # Check what files would be processed
        zip_names = set(file_name for file_name, index in data)
        print(f"Unique zip files: {len(zip_names)}")
        
        for zip_name in list(zip_names)[:3]:  # Show first 3
            base_name = zip_name.replace('.tar.gz', '')
            data_file_name = base_name + data_type
            chunk_file = os.path.join(processor.data_dir, data_file_name + '_features_0.npy')
            print(f"  {zip_name} -> {data_file_name}")
            print(f"    Chunk file exists: {os.path.exists(chunk_file)}")
    else:
        print(f"ERROR: No samples drawn for {data_type}!")
    
    return data

def main():
    # Configuration
    go_board_rows, go_board_cols = 19, 19
    num_classes = go_board_rows * go_board_cols
    learning_rate = 0.01
    epochs = 50
    
    encoder = SimpleEncoder((go_board_rows, go_board_cols))
    processor = GoDataProcessor(encoder=encoder.name())
    
    # Load generators (this processes and saves data to .npy files)
    print("Loading training data...")
    train_generator = processor.load_go_data('train', 10000, use_generator=True)
    
    print("Loading test data...")
    test_generator = processor.load_go_data('test', 1000, use_generator=True)
    
    # Get sample COUNTS (integers, not lists)
    train_samples = train_generator.get_num_samples()  # Returns int
    test_samples = test_generator.get_num_samples()    # Returns int
    
    print(f"Training samples: {train_samples}")
    print(f"Validation samples: {test_samples}")
    
    if train_samples == 0:
        raise ValueError("No training samples found!")
    if test_samples == 0:
        raise ValueError("No test samples found!")
    
    # Continue with training
    batch_size = 128
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Now train_samples and test_samples are integers
    train_loader = GeneratorDataLoader(train_generator, train_samples, batch_size, num_classes)
    val_loader = GeneratorDataLoader(test_generator, test_samples, batch_size, num_classes)
    
    # Create model
    model = GoNetwork.GoNetwork(input_channels=encoder.num_planes, num_classes=num_classes)
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
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {max(history['val_acc']):.4f}")
    print("\nTo view training in TensorBoard, run:")
    print("  tensorboard --logdir ../checkpoints/logs")