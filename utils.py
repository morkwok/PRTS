import os
import numpy as np
import cv2
import json
import h5py
import torch
from PIL import Image
import pickle

def load_image(image_path):
    """Load an image from the specified path"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise Exception(f"Failed to load image: {image_path}")
        # Convert BGR to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None


def save_image(image, save_path):
    """Save an image to the specified path"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Convert RGB to BGR before saving with cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, image)
        return True
    except Exception as e:
        print(f"Error saving image {save_path}: {e}")
        return False


def load_json_file(file_path):
    """Load JSON data from the specified file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file {file_path}: {e}")
        return None


def save_json_file(data, file_path):
    """Save data to the specified JSON file"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Error saving JSON file {file_path}: {e}")
        return False


def load_h5_file(file_path):
    """Load data from an HDF5 file"""
    try:
        data = {}
        with h5py.File(file_path, 'r') as f:
            def visit_item(name, node):
                if isinstance(node, h5py.Dataset):
                    data[name] = node[()]
            f.visititems(visit_item)
        return data
    except Exception as e:
        print(f"Error loading HDF5 file {file_path}: {e}")
        return None


def save_h5_file(data, file_path):
    """Save data to an HDF5 file"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with h5py.File(file_path, 'w') as f:
            for key, value in data.items():
                # Handle different data types
                if isinstance(value, np.ndarray):
                    f.create_dataset(key, data=value)
                elif isinstance(value, (int, float, str, bytes)):
                    f[key] = value
                elif isinstance(value, (list, tuple)):
                    f.create_dataset(key, data=np.array(value))
                else:
                    # Convert other types to strings
                    f[key] = str(value)
        return True
    except Exception as e:
        print(f"Error saving HDF5 file {file_path}: {e}")
        return False


def load_pickle_file(file_path):
    """Load data from a pickle file"""
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pickle file {file_path}: {e}")
        return None


def save_pickle_file(data, file_path):
    """Save data to a pickle file"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        return True
    except Exception as e:
        print(f"Error saving pickle file {file_path}: {e}")
        return False


def resize_image(image, target_size, interpolation=cv2.INTER_LINEAR):
    """Resize image to the target size"""
    try:
        resized_image = cv2.resize(image, target_size, interpolation=interpolation)
        return resized_image
    except Exception as e:
        print(f"Error resizing image: {e}")
        return None


def crop_image(image, crop_coords):
    """Crop image using the specified coordinates (x1, y1, x2, y2)"""
    try:
        x1, y1, x2, y2 = crop_coords
        cropped_image = image[y1:y2, x1:x2]
        return cropped_image
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None


def normalize_image(image, mean=None, std=None):
    """Normalize image pixel values"""
    try:
        normalized_image = image.astype(np.float32)
        
        if mean is not None and std is not None:
            # Normalize using provided mean and std
            normalized_image = (normalized_image - mean) / std
        else:
            # Simple normalization to [0, 1]
            normalized_image = normalized_image / 255.0
            
        return normalized_image
    except Exception as e:
        print(f"Error normalizing image: {e}")
        return None


def get_mask_area(mask):
    """Calculate the area of the mask (number of non-zero pixels)"""
    try:
        return np.count_nonzero(mask)
    except Exception as e:
        print(f"Error calculating mask area: {e}")
        return 0


def get_bounding_box(mask):
    """Get the bounding box coordinates of the mask (x1, y1, x2, y2)"""
    try:
        # Find coordinates of non-zero pixels
        coords = np.where(mask > 0)
        if len(coords[0]) == 0 or len(coords[1]) == 0:
            return (0, 0, 0, 0)
        
        y_min, y_max = np.min(coords[0]), np.max(coords[0])
        x_min, x_max = np.min(coords[1]), np.max(coords[1])
        
        return (x_min, y_min, x_max, y_max)
    except Exception as e:
        print(f"Error getting bounding box: {e}")
        return (0, 0, 0, 0)


def draw_bounding_box(image, bbox, color=(255, 0, 0), thickness=2):
    """Draw a bounding box on the image"""
    try:
        x1, y1, x2, y2 = bbox
        image_with_box = image.copy()
        cv2.rectangle(image_with_box, (x1, y1), (x2, y2), color, thickness)
        return image_with_box
    except Exception as e:
        print(f"Error drawing bounding box: {e}")
        return image


def calculate_rmse(prediction, target):
    """Calculate Root Mean Squared Error between prediction and target"""
    try:
        return np.sqrt(np.mean((prediction - target) ** 2))
    except Exception as e:
        print(f"Error calculating RMSE: {e}")
        return None


def calculate_pcc(prediction, target):
    """Calculate Pearson Correlation Coefficient between prediction and target"""
    try:
        # Check if inputs are 1D arrays
        if prediction.ndim > 1:
            prediction = prediction.flatten()
        if target.ndim > 1:
            target = target.flatten()
            
        # Calculate Pearson correlation coefficient
        return np.corrcoef(prediction, target)[0, 1]
    except Exception as e:
        print(f"Error calculating PCC: {e}")
        return None


def create_directory(directory_path):
    """Create a directory if it doesn't exist"""
    try:
        os.makedirs(directory_path, exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating directory {directory_path}: {e}")
        return False


def list_files(directory_path, extension=None):
    """List all files in the directory with the specified extension"""
    try:
        files = []
        for file in os.listdir(directory_path):
            file_path = os.path.join(directory_path, file)
            if os.path.isfile(file_path):
                if extension is None or file.endswith(extension):
                    files.append(file_path)
        return files
    except Exception as e:
        print(f"Error listing files in {directory_path}: {e}")
        return []


def convert_tensor_to_numpy(tensor):
    """Convert a PyTorch tensor to a NumPy array"""
    try:
        if tensor.requires_grad:
            tensor = tensor.detach()
        if tensor.is_cuda:
            tensor = tensor.cpu()
        return tensor.numpy()
    except Exception as e:
        print(f"Error converting tensor to numpy array: {e}")
        return None


def convert_numpy_to_tensor(array):
    """Convert a NumPy array to a PyTorch tensor"""
    try:
        return torch.from_numpy(array)
    except Exception as e:
        print(f"Error converting numpy array to tensor: {e}")
        return None


def apply_log1p_transform(data):
    """Apply log1p transformation to the data"""
    try:
        return np.log1p(data)
    except Exception as e:
        print(f"Error applying log1p transformation: {e}")
        return None


def apply_expm1_transform(data):
    """Apply expm1 transformation to the data (inverse of log1p)"""
    try:
        return np.expm1(data)
    except Exception as e:
        print(f"Error applying expm1 transformation: {e}")
        return None


def normalize_features(features, mean=None, std=None):
    """Normalize features using z-score normalization"""
    try:
        if mean is None:
            mean = np.mean(features, axis=0)
        if std is None:
            std = np.std(features, axis=0)
            # Avoid division by zero
            std[std == 0] = 1.0
        
        normalized_features = (features - mean) / std
        return normalized_features, mean, std
    except Exception as e:
        print(f"Error normalizing features: {e}")
        return None, None, None


def load_model(model_path, device=None):
    """Load a PyTorch model from the specified path"""
    try:
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = torch.load(model_path, map_location=device)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None


def save_model(model, model_path):
    """Save a PyTorch model to the specified path"""
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(model, model_path)
        return True
    except Exception as e:
        print(f"Error saving model {model_path}: {e}")
        return False


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=None):
    """Split data into training, validation, and test sets"""
    try:
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Check if ratios sum to 1
        assert np.isclose(train_ratio + val_ratio + test_ratio, 1.0), "Ratios must sum to 1"
        
        # Get data indices
        num_samples = len(data)
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(train_ratio * num_samples)
        val_end = train_end + int(val_ratio * num_samples)
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Create data splits
        train_data = [data[i] for i in train_indices]
        val_data = [data[i] for i in val_indices]
        test_data = [data[i] for i in test_indices]
        
        return train_data, val_data, test_data
    except Exception as e:
        print(f"Error splitting data: {e}")
        return None, None, None