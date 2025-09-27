import os
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from einops import rearrange
import timm

class ViTFeatureExtractor:
    """
    Vision Transformer feature extractor for histological images
    Uses a pre-trained ViT-256/16 model trained with DINO
    """
    def __init__(self, model_path=None, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        
        # Model parameters
        self.patch_size = 16
        self.image_size = 256
        self.hidden_dim = 384  # DINO ViT-Small has hidden dimension of 384
        
    def _load_model(self, model_path=None):
        """
        Load pre-trained ViT model
        Args:
            model_path: Optional path to custom model weights
        Returns:
            Loaded ViT model
        """
        if model_path is not None and os.path.exists(model_path):
            print(f"Loading custom model from {model_path}")
            # For custom model weights
            model = timm.create_model('vit_small_patch16_256', pretrained=False)
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
        else:
            print("Loading pre-trained ViT-Small (patch16_256) with DINO weights")
            # Use timm to load the model with DINO weights
            model = timm.create_model('vit_small_patch16_256', pretrained=True)
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self):
        """
        Get image transformation pipeline
        Returns:
            Composition of image transforms
        """
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess a single image for ViT input
        Args:
            image: PIL Image or numpy array
        Returns:
            Preprocessed tensor ready for model input
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transformations
        transformed = self.transform(image).unsqueeze(0)  # Add batch dimension
        return transformed.to(self.device)
    
    def extract_features(self, image, extract_patches=False):
        """
        Extract features from an image using ViT
        Args:
            image: PIL Image or numpy array
            extract_patches: Whether to return patch features along with CLS token
        Returns:
            Dictionary containing extracted features
        """
        # Preprocess image
        input_tensor = self.preprocess_image(image)
        
        with torch.no_grad():
            # Forward pass through the model
            # Get the outputs before the final classification layer
            features = self.model.forward_features(input_tensor)
            
            # Extract CLS token feature (global feature)
            cls_token = features[:, 0, :].squeeze(0).cpu().numpy()
            
            # Extract patch tokens (local features)
            patch_tokens = features[:, 1:, :].squeeze(0).cpu().numpy()
        
        result = {
            'cls_token': cls_token,  # Global feature
            'patch_tokens': patch_tokens  # Local features
        }
        
        return result
    
    def get_patch_coordinates(self, cell_center, patch_size=16, image_size=256):
        """
        Calculate coordinates of patches that overlap with the cell nucleus
        Args:
            cell_center: (x, y) coordinates of the cell center
            patch_size: Size of each patch in pixels
            image_size: Size of the image in pixels
        Returns:
            List of patch indices that overlap with the cell
        """
        # Calculate number of patches in each dimension
        num_patches = image_size // patch_size
        
        # Convert cell center to patch indices
        cell_patch_x = cell_center[0] // patch_size
        cell_patch_y = cell_center[1] // patch_size
        
        # Define a region around the cell center (e.g., 3x3 patches)
        patch_indices = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                px = cell_patch_x + dx
                py = cell_patch_y + dy
                if 0 <= px < num_patches and 0 <= py < num_patches:
                    patch_idx = px + py * num_patches
                    patch_indices.append(patch_idx)
        
        return patch_indices
    
    def extract_cell_features(self, image, cell_center=None):
        """
        Extract combined features for a single cell
        Args:
            image: 256x256 pixel image centered on the cell
            cell_center: (x, y) coordinates of the cell center within the image
        Returns:
            Dictionary containing global, local, and fused features
        """
        # Extract features from the image
        features = self.extract_features(image, extract_patches=True)
        
        # Get global feature from CLS token
        global_feature = features['cls_token']  # Shape: (384,)
        
        # If cell center is provided, get local features from patches around the center
        if cell_center is not None:
            # Calculate which patches overlap with the cell nucleus
            patch_indices = self.get_patch_coordinates(cell_center)
            
            # Extract relevant patch features
            local_patch_features = features['patch_tokens'][patch_indices]
            
            # Average pooling to get a single local feature vector
            local_feature = np.mean(local_patch_features, axis=0)  # Shape: (384,)
        else:
            # If no cell center is provided, use all patch features
            local_feature = np.mean(features['patch_tokens'], axis=0)  # Shape: (384,)
        
        # Fuse global and local features by concatenation
        # According to the technical description: global_feature (384) + local_feature (192) = 576 dimensions
        # We need to reduce local_feature from 384 to 192 dimensions
        local_feature_reduced = local_feature[:192]  # Take first 192 dimensions
        fused_feature = np.concatenate([global_feature, local_feature_reduced])  # Shape: (576,)
        
        return {
            'global': global_feature,  # Shape: (384,)
            'local': local_feature_reduced,  # Shape: (192,)
            'fused': fused_feature  # Shape: (576,)
        }

class BatchProcessor:
    """
    Batch processor for images to extract features in parallel
    """
    def __init__(self, feature_extractor, batch_size=32):
        self.feature_extractor = feature_extractor
        self.batch_size = batch_size
    
    def process_batch(self, images, cell_centers=None):
        """
        Process a batch of images
        Args:
            images: List of PIL Images or numpy arrays
            cell_centers: Optional list of cell center coordinates
        Returns:
            Dictionary of features for each image in the batch
        """
        results = []
        
        # Process in batches
        for i in range(0, len(images), self.batch_size):
            batch_images = images[i:i+self.batch_size]
            
            if cell_centers is not None:
                batch_centers = cell_centers[i:i+self.batch_size]
            else:
                batch_centers = [None] * len(batch_images)
            
            # Process each image in the batch
            for img, center in zip(batch_images, batch_centers):
                features = self.feature_extractor.extract_cell_features(img, center)
                results.append(features)
        
        return results

def convert_patches_to_image(patch_features, patch_size=16, image_size=256):
    """
    Convert patch features back to an image-like representation
    Args:
        patch_features: Patch features array of shape (num_patches, feature_dim)
        patch_size: Size of each patch in pixels
        image_size: Size of the output image in pixels
    Returns:
        Image-like array representation of the patch features
    """
    # Calculate number of patches in each dimension
    num_patches = image_size // patch_size
    
    # Reshape patch features to 2D grid
    feature_grid = rearrange(patch_features, '(h w) c -> h w c', h=num_patches, w=num_patches)
    
    return feature_grid

def normalize_features(features, mean=None, std=None):
    """
    Normalize features using provided or computed mean and standard deviation
    Args:
        features: Input feature array
        mean: Optional mean for normalization
        std: Optional standard deviation for normalization
    Returns:
        Normalized features array
    """
    if mean is None:
        mean = np.mean(features, axis=0)
    if std is None:
        std = np.std(features, axis=0) + 1e-8  # Add epsilon to avoid division by zero
    
    normalized_features = (features - mean) / std
    return normalized_features

def save_model(model, save_path, config=None):
    """
    Save model weights and configuration
    Args:
        model: PyTorch model
        save_path: Path to save the model
        config: Optional configuration dictionary
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    save_dict = {
        'state_dict': model.state_dict()
    }
    
    if config is not None:
        save_dict['config'] = config
    
    torch.save(save_dict, save_path)
    print(f"Model saved to {save_path}")

def load_model(model_class, save_path, device='cuda'):
    """
    Load model weights and configuration
    Args:
        model_class: Class of the model to load
        save_path: Path to the saved model
        device: Device to load the model onto
    Returns:
        Loaded model and configuration
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(save_path, map_location=device)
    
    # Extract configuration if available
    config = checkpoint.get('config', {})
    
    # Create model instance
    model = model_class(**config)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def calculate_rescale_factor(current_pixel_size, target_pixel_size=0.5):
    """
    Calculate the rescaling factor to achieve the target pixel size
    Args:
        current_pixel_size: Current pixel size in μm
        target_pixel_size: Target pixel size in μm
    Returns:
        Rescaling factor
    """
    return current_pixel_size / target_pixel_size

def resize_image(image, scale_factor):
    """
    Resize an image using the given scale factor
    Args:
        image: PIL Image or numpy array
        scale_factor: Rescaling factor
    Returns:
        Resized image
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    width, height = image.size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    resized_image = image.resize((new_width, new_height), Image.BILINEAR)
    
    return resized_image