from time import time
import argparse
import pickle
import os

from einops import rearrange, reduce, repeat
import numpy as np
import skimage
import torch
import pandas as pd
import anndata as ad

from utils import load_image
from utils import load_pickle, save_pickle, join
from PIL import Image
from einops import rearrange

import torch.multiprocessing
from torchvision import transforms

from model_utils import (
    get_vit256,
    tensorbatch2im,
    eval_transforms
)

def load_cell_ids(prefix, h5ad_file=None):
    """
    Load cell IDs either from h5ad file or image directory
    Args:
        prefix: Directory prefix
        h5ad_file: Optional h5ad file path
    Returns:
        List of cell IDs
    """
    if h5ad_file:
        print(f"Reading cell IDs from h5ad file: {h5ad_file}")
        adata = ad.read_h5ad(h5ad_file)
        cell_ids = adata.obs.index.tolist()
    else:
        print(f"Reading all cell IDs from image directory: {prefix}cells_256/")
        cells_dir = f'{prefix}cells_256/'
        if not os.path.exists(cells_dir):
            raise FileNotFoundError(f"Directory not found: {cells_dir}")
        
        cell_ids = []
        files = os.listdir(cells_dir)
        for file in files:
            if file.endswith('_256.tif'):
                cell_id = file.replace('_256.tif', '')
                cell_ids.append(cell_id)
        
        print(f"Found {len(cell_ids)} cell images in {cells_dir}")
    
    return cell_ids

def get_data_batch(prefix, cell_ids, batch_idx, batch_size):
    """
    Load a batch of cell images
    Args:
        prefix: Directory prefix
        cell_ids: List of cell IDs
        batch_idx: Current batch index
        batch_size: Batch size
    Returns:
        Tuple of (image batch array, batch cell IDs)
    """
    start_idx = batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(cell_ids))
    batch_cell_ids = cell_ids[start_idx:end_idx]

    embs_256 = [] 
    
    print(f"Loading batch {batch_idx+1} of cell images ({start_idx}:{end_idx})...")
    for cell_id in batch_cell_ids:
        global_path = f'{prefix}/{cell_id}_256.tif'
        
        if not os.path.exists(global_path):
            print(f"Warning: Image file for cell {cell_id} does not exist, skipping")
            continue

        img_256 = load_image(global_path)
        img_256 = img_256.astype(np.float32) / 255.0
        embs_256.append(img_256)

    if len(embs_256) == 0:
        raise ValueError(f"No valid cell images found in batch {batch_idx}")
        
    embs_256 = np.stack(embs_256)  
    
    return embs_256, batch_cell_ids

def extract_and_process_features(model, embs_256, args):
    """
    Extract and process features from images using ViT model
    Args:
        model: Pretrained ViT model
        embs_256: Batch of 256x256 images
        args: Command line arguments
    Returns:
        Dictionary of extracted features
    """
    model = model.to(args.device)
    model.eval()
    
    features = {}
    with torch.no_grad():
        imgs = torch.stack([eval_transforms()(img) for img in embs_256]).to(args.device)
        fea_all256 = model.forward_all(imgs).cpu()
        
        # Extract global features from CLS token
        cls_features = fea_all256[:, 0].numpy()  # Global feature vector (192 dimensions)
        
        # Extract local features from patches overlapping with cell nucleus
        # Reshape to (batch, 16, 16, 384) to represent the 16x16 patches
        sub_feat = fea_all256[:, 1:].reshape(-1, 16, 16, 384)
        
        # In a real implementation, we would determine which patches overlap with the cell nucleus
        # For this example, we'll use the center 2x2 patches as a simple approximation
        tart_h = (16 - 2) // 2
        start_w = (16 - 2) // 2
        middle_patches = sub_feat[:, tart_h:tart_h+2, start_w:start_w+2, :]
        
        # Average pooling to aggregate patch features
        middle_patches = rearrange(middle_patches, 'b h w c -> b c h w')
        sub_features = reduce(middle_patches, 'b c h w -> b c', 'mean')
        
        # Local feature vector (384 dimensions)
        sub_features = sub_features.numpy()
        
        # Concatenate global and local features (192 + 384 = 576 dimensions)
        fused_features = np.concatenate((cls_features, sub_features), axis=1)
        
        features['global'] = cls_features  # Shape: (batch_size, 192)
        features['local'] = sub_features   # Shape: (batch_size, 384)
        features['fused'] = fused_features # Shape: (batch_size, 576)

    print("Feature extraction completed.")
    print(f"Global features shape: {features['global'].shape} - (number of cells, 192)")
    print(f"Local features shape: {features['local'].shape} - (number of cells, 384)")
    print(f"Fused features shape: {features['fused'].shape} - (number of cells, 576)")
    
    return features

def parse_args():
    """\Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract features from cell images')
    parser.add_argument('--prefix', type=str, required=True, help='Directory prefix for cell images')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--output_file', type=str, default='features.pkl', help='Output file path')
    parser.add_argument('--h5ad_file', type=str, default=None, help='Optional h5ad file for cell IDs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for computation')
    return parser.parse_args()

def main():
    """Main function to extract features from cell images"""
    args = parse_args()
    
    # Load cell IDs
    cell_ids = load_cell_ids(args.prefix, args.h5ad_file)
    
    # Initialize model
    print("Loading pre-trained ViT-256/16 model...")
    model = get_vit256(pretrained_weights=None, arch='vit_small', device=args.device)
    
    # Process in batches
    all_features = {}
    total_batches = (len(cell_ids) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(total_batches):
        try:
            # Get batch data
            batch_images, batch_cell_ids = get_data_batch(args.prefix, cell_ids, batch_idx, args.batch_size)
            
            # Extract features
            batch_features = extract_and_process_features(model, batch_images, args)
            
            # Store results
            for i, cell_id in enumerate(batch_cell_ids):
                all_features[cell_id] = {
                    'global': batch_features['global'][i],
                    'local': batch_features['local'][i],
                    'fused': batch_features['fused'][i]
                }
            
            # Print progress
            progress = (batch_idx + 1) / total_batches * 100
            print(f"Progress: {progress:.1f}% ({batch_idx + 1}/{total_batches} batches processed)")
            
        except Exception as e:
            print(f"Error processing batch {batch_idx}: {e}")
            continue
    
    # Save features
    print(f"Saving extracted features to {args.output_file}...")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with open(args.output_file, 'wb') as f:
        pickle.dump({
            'cell_ids': list(all_features.keys()),
            'features': {
                'global': np.array([all_features[cell]['global'] for cell in all_features]),
                'local': np.array([all_features[cell]['local'] for cell in all_features]),
                'fused': np.array([all_features[cell]['fused'] for cell in all_features])
            }
        }, f)
    
    print(f"Feature extraction completed for {len(all_features)} cells")

if __name__ == '__main__':
    main()