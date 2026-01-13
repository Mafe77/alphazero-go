import os
import random
import json
from pathlib import Path
from typing import List, Tuple, Set, Optional


class Sampler:
    """
    Sample training and test data from zipped SGF files with stable test set.
    
    Improvements:
    - Better file handling with pathlib
    - JSON format for test samples (more robust than eval)
    - Caching of available games to avoid recomputing
    - Type hints for better code documentation
    - More efficient set operations
    - Better error handling
    """
    
    def __init__(self, data_dir: str = 'data', num_test_games: int = 100, 
                 cap_year: int = 2015, seed: int = 1337):
        """
        Initialize sampler.
        
        Args:
            data_dir: Directory containing SGF data
            num_test_games: Number of games to reserve for testing
            cap_year: Maximum year to include (for stability)
            seed: Random seed for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.num_test_games = num_test_games
        self.cap_year = cap_year
        self.seed = seed
        
        # Use JSON instead of Python literals (safer than eval)
        self.test_file = self.data_dir / 'test_samples.json'
        
        # Cache for available games
        self._available_games_cache: Optional[List[Tuple[str, int]]] = None
        
        # Initialize test and train games
        self.test_games: Set[Tuple[str, int]] = set()
        self.train_games: List[Tuple[str, int]] = []
        
        random.seed(seed)
        self._load_or_create_test_samples()
    
    def draw_data(self, data_type: str, num_samples: Optional[int] = None) -> List[Tuple[str, int]]:
        """
        Draw data samples based on type.
        
        Args:
            data_type: 'train' or 'test'
            num_samples: Number of samples to draw (only for train)
            
        Returns:
            List of (filename, game_index) tuples
        """
        if data_type == 'test':
            return list(self.test_games)
        elif data_type == 'train':
            if num_samples is None:
                return self._draw_all_training()
            else:
                return self._draw_training_samples(num_samples)
        else:
            raise ValueError(f"Invalid data_type '{data_type}'. Choose 'train' or 'test'.")
    
    def _get_available_games(self, exclude_test: bool = True) -> List[Tuple[str, int]]:
        """
        Get all available games up to cap_year.
        
        Args:
            exclude_test: If True, exclude test games from results
            
        Returns:
            List of (filename, game_index) tuples
        """
        # Use cache if available
        if self._available_games_cache is None:
            from dlgo.data.index_processor import KGSIndex
            
            available_games = []
            index = KGSIndex(data_directory=str(self.data_dir))
            
            for fileinfo in index.file_info:
                filename = fileinfo['filename']
                
                # Parse year from filename (format: KGS-YYYY-...)
                try:
                    year = int(filename.split('-')[1].split('_')[0])
                except (IndexError, ValueError):
                    print(f"Warning: Could not parse year from {filename}, skipping")
                    continue
                
                if year > self.cap_year:
                    continue
                
                num_games = fileinfo.get('num_games', 0)
                for i in range(num_games):
                    available_games.append((filename, i))
            
            self._available_games_cache = available_games
            print(f'Total available games (year ≤ {self.cap_year}): {len(available_games)}')
        
        if exclude_test:
            # Filter out test games
            return [game for game in self._available_games_cache if game not in self.test_games]
        else:
            return self._available_games_cache.copy()
    
    def _load_or_create_test_samples(self):
        """Load existing test samples or create new ones."""
        if self.test_file.exists():
            # Load existing test samples
            with open(self.test_file, 'r') as f:
                test_data = json.load(f)
            
            self.test_games = set((item['filename'], item['index']) for item in test_data)
            print(f'Loaded {len(self.test_games)} test samples from {self.test_file}')
        else:
            # Create new test samples
            print(f'Creating {self.num_test_games} test samples...')
            self._create_test_samples()
    
    def _create_test_samples(self):
        """Create and save a fixed set of test samples."""
        available_games = self._get_available_games(exclude_test=False)
        
        if len(available_games) < self.num_test_games:
            raise ValueError(
                f"Not enough games available ({len(available_games)}) "
                f"to create {self.num_test_games} test samples"
            )
        
        # Randomly sample test games
        test_games_list = random.sample(available_games, self.num_test_games)
        self.test_games = set(test_games_list)
        
        # Save to file in JSON format
        test_data = [
            {'filename': filename, 'index': idx} 
            for filename, idx in sorted(test_games_list)
        ]
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.test_file, 'w') as f:
            json.dump(test_data, f, indent=2)
        
        print(f'Created and saved {len(self.test_games)} test samples to {self.test_file}')
    
    def _draw_training_samples(self, num_samples: int) -> List[Tuple[str, int]]:
        """
        Draw random training samples, excluding test games.
        
        Args:
            num_samples: Number of training samples to draw
            
        Returns:
            List of (filename, game_index) tuples
        """
        available_train_games = self._get_available_games(exclude_test=True)
        
        if len(available_train_games) < num_samples:
            print(
                f"Warning: Requested {num_samples} samples but only "
                f"{len(available_train_games)} available. Using all available."
            )
            num_samples = len(available_train_games)
        
        samples = random.sample(available_train_games, num_samples)
        print(f'Drew {len(samples)} training samples')
        return samples
    
    def _draw_all_training(self) -> List[Tuple[str, int]]:
        """
        Draw all available training games (excluding test set).
        
        Returns:
            List of (filename, game_index) tuples
        """
        all_train_games = self._get_available_games(exclude_test=True)
        print(f'Drew all training samples: {len(all_train_games)} games')
        return all_train_games
    
    def get_statistics(self) -> dict:
        """Get statistics about the dataset."""
        available_games = self._get_available_games(exclude_test=False)
        train_games = self._get_available_games(exclude_test=True)
        
        return {
            'total_games': len(available_games),
            'test_games': len(self.test_games),
            'available_train_games': len(train_games),
            'cap_year': self.cap_year,
            'test_file': str(self.test_file),
            'test_file_exists': self.test_file.exists(),
        }
    
    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()
        print("\n" + "="*50)
        print("Dataset Statistics")
        print("="*50)
        print(f"Total games (year ≤ {stats['cap_year']}): {stats['total_games']}")
        print(f"Test games: {stats['test_games']}")
        print(f"Available training games: {stats['available_train_games']}")
        print(f"Test file: {stats['test_file']}")
        print(f"Test file exists: {stats['test_file_exists']}")
        print("="*50 + "\n")
    
    def reset_test_samples(self):
        """Delete test samples file and regenerate."""
        if self.test_file.exists():
            self.test_file.unlink()
            print(f"Deleted {self.test_file}")
        
        self.test_games = set()
        self._create_test_samples()
