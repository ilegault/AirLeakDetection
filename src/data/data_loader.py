import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import yaml


class WebDAQDataLoader:
    """Load and parse WebDAQ CSV files"""
    
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_single_file(self, filepath: Path) -> np.ndarray:
        """
        Load a single CSV file from WebDAQ
        Returns: numpy array of shape (n_samples, n_channels)
        """
        # Read CSV - adjust based on your actual CSV structure
        df = pd.read_csv(filepath)
        
        # Assuming columns are: Time, Accel1_X, Accel1_Y, Accel1_Z, etc.
        # Skip time column and get accelerometer data
        data = df.iloc[:, 1:10].values  # 9 channels
        
        return data
    
    def load_dataset(self) -> Dict[str, np.ndarray]:
        """Load all data files and organize by class"""
        data_dict = {}
        base_path = Path(self.config['data']['raw_data_path'])
        
        for class_name in ['NOLEAK', 'SMALL_0.125', 'MEDIUM_0.25', 'LARGE_0.5']:
            class_path = base_path / class_name
            if class_path.exists():
                files = list(class_path.glob('*.csv'))
                class_data = []
                
                for file in files:
                    data = self.load_single_file(file)
                    class_data.append(data)
                
                data_dict[class_name] = np.array(class_data)
        
        return data_dict