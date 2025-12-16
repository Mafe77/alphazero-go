import glob
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader


class DataGenerator:
    """
    PyTorch-compatible data generator for Go training data.
    Returns integer labels instead of one-hot (PyTorch standard).
    """
    
    def __init__(self, data_directory, samples, shuffle=True, seed=None):
        self.data_directory = data_directory
        self.samples = samples
        self.files = sorted(set(file_name for file_name, index in samples))
        self.shuffle = shuffle
        self.seed = seed
        self.num_samples = None
        self.rng = np.random.default_rng(seed)

    def get_num_samples(self, batch_size=128, num_classes=19 * 19):
        """Efficiently count samples by checking file shapes without loading all data."""
        if self.num_samples is not None:
            return self.num_samples
        
        self.num_samples = 0
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + 'train'
            base = os.path.join(self.data_directory, file_name + '_features_*.npy')
            
            for feature_file in glob.glob(base):
                if not os.path.exists(feature_file):
                    continue
                # Use mmap_mode to check shape without loading into memory
                x = np.load(feature_file, mmap_mode='r')
                self.num_samples += x.shape[0]
        
        return self.num_samples

    def _generate(self, batch_size, num_classes, return_tensors=True):
        """
        Internal generator that yields batches once through the dataset.
        
        Args:
            batch_size: Number of samples per batch
            num_classes: Number of output classes (361 for 19x19 Go)
            return_tensors: If True, return PyTorch tensors. If False, return numpy arrays.
        """
        files = self.files.copy()
        if self.shuffle:
            self.rng.shuffle(files)
        
        for zip_file_name in files:
            file_name = zip_file_name.replace('.tar.gz', '') + 'train'
            base = os.path.join(self.data_directory, file_name + '_features_*.npy')
            feature_files = sorted(glob.glob(base))
            
            if self.shuffle:
                self.rng.shuffle(feature_files)
            
            for feature_file in feature_files:
                label_file = feature_file.replace('features', 'labels')
                
                # Validate files exist
                if not os.path.exists(feature_file):
                    print(f"Warning: Feature file not found: {feature_file}")
                    continue
                if not os.path.exists(label_file):
                    print(f"Warning: Label file not found: {label_file}")
                    continue
                
                try:
                    x = np.load(feature_file)
                    y = np.load(label_file)
                except Exception as e:
                    print(f"Error loading {feature_file}: {e}")
                    continue
                
                # Validate shapes match
                if x.shape[0] != y.shape[0]:
                    print(f"Warning: Shape mismatch in {feature_file}: x={x.shape[0]}, y={y.shape[0]}")
                    continue
                
                # Remove this later for testing with pytorch
                x = np.transpose(x, (0, 2, 3, 1))  # (N, C, H, W) -> (N, H, W, C)
                x = x.astype('float32')
                
                # keeping labels as ints for pytorch
                y = y.astype(np.int64)
                
                # Shuffle samples within this file
                if self.shuffle:
                    indices = self.rng.permutation(x.shape[0])
                    x = x[indices]
                    y = y[indices]
                
                # Yield full batches
                num_full_batches = x.shape[0] // batch_size
                for i in range(num_full_batches):
                    start_idx = i * batch_size
                    end_idx = start_idx + batch_size
                    x_batch = x[start_idx:end_idx]
                    y_batch = y[start_idx:end_idx]
                    
                    if return_tensors:
                        # Convert to PyTorch tensors and transpose to NCHW format
                        x_batch = np.transpose(x_batch, (0, 3, 1, 2))  # NHWC -> NCHW
                        x_tensor = torch.from_numpy(x_batch).float()
                        y_tensor = torch.from_numpy(y_batch).long()
                        yield x_tensor, y_tensor
                    else:
                        yield x_batch, y_batch
                
                # Yield remaining partial batch
                remainder = x.shape[0] % batch_size
                if remainder > 0:
                    x_batch = x[-remainder:]
                    y_batch = y[-remainder:]
                    
                    if return_tensors:
                        x_batch = np.transpose(x_batch, (0, 3, 1, 2))  # NHWC -> NCHW
                        x_tensor = torch.from_numpy(x_batch).float()
                        y_tensor = torch.from_numpy(y_batch).long()
                        yield x_tensor, y_tensor
                    else:
                        yield x_batch, y_batch

    def generate(self, batch_size=128, num_classes=19 * 19, return_tensors=True):
        """
        Infinite generator for training. Loops through dataset indefinitely.
        Reshuffles between epochs if shuffle=True.
        
        Args:
            batch_size: Number of samples per batch
            num_classes: Number of output classes
            return_tensors: If True, return PyTorch tensors. If False, return numpy arrays.
        """
        while True:
            for item in self._generate(batch_size, num_classes, return_tensors):
                yield item

    def generate_once(self, batch_size=128, num_classes=19 * 19, return_tensors=True):
        """
        Single-pass generator for validation/testing.
        Useful when you want to evaluate on the full dataset once.
        """
        for item in self._generate(batch_size, num_classes, return_tensors):
            yield item


class GoDataset(Dataset):
    """
    PyTorch Dataset wrapper for Go data.
    Loads all data into memory at initialization.
    Use this if your dataset fits in memory for better performance.
    """
    
    def __init__(self, data_directory, samples, num_classes=19 * 19):
        self.data_directory = data_directory
        self.samples = samples
        self.files = sorted(set(file_name for file_name, index in samples))
        self.num_classes = num_classes
        
        # Load all data
        self.features, self.labels = self._load_all_data()
        
    def _load_all_data(self):
        """Load all data files into memory."""
        feature_list = []
        label_list = []
        
        for zip_file_name in self.files:
            file_name = zip_file_name.replace('.tar.gz', '') + 'train'
            base = os.path.join(self.data_directory, file_name + '_features_*.npy')
            
            for feature_file in sorted(glob.glob(base)):
                label_file = feature_file.replace('features', 'labels')
                
                if not os.path.exists(feature_file) or not os.path.exists(label_file):
                    continue
                
                try:
                    x = np.load(feature_file)
                    y = np.load(label_file)
                    
                    # Transpose to PyTorch format: (N, C, H, W)
                    # Already in this format from file, so no transpose needed
                    x = x.astype('float32')
                    y = y.astype(np.int64)
                    
                    feature_list.append(x)
                    label_list.append(y)
                except Exception as e:
                    print(f"Error loading {feature_file}: {e}")
                    continue
        
        if not feature_list:
            raise ValueError("No data files found!")
        
        features = np.concatenate(feature_list, axis=0)
        labels = np.concatenate(label_list, axis=0)
        
        return features, labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        """Get a single sample."""
        x = torch.from_numpy(self.features[idx]).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def create_dataloader(data_directory, samples, batch_size=128, shuffle=True, 
                     num_workers=0, use_dataset=False):
    """
    Factory function to create a PyTorch DataLoader.
    
    Args:
        data_directory: Directory containing data files
        samples: List of (filename, index) tuples
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes for data loading
        use_dataset: If True, use GoDataset (loads all in memory). 
                    If False, use DataGenerator (streaming).
    
    Returns:
        DataLoader or DataGenerator instance
    """
    if use_dataset:
        # Load all data in memory - faster if data fits in RAM
        dataset = GoDataset(data_directory, samples)
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    else:
        # Use generator - better for large datasets
        return DataGenerator(data_directory, samples, shuffle=shuffle)
