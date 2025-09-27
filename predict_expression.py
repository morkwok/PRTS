import os
import time
import argparse
import pickle
import numpy as np
import torch
import anndata as ad
from sklearn.preprocessing import StandardScaler

from train_model import DualOutputMLP, CustomPytorchRegressor

class Predictor:
    """
    Class for gene expression prediction using trained model
    Args:
        model_path: Path to the trained model weights
        features_file: Path to the features file
        input_dim: Input feature dimension
        output_dim: Output dimension (number of genes)
    """
    def __init__(self, model_path, features_file, input_dim=None, output_dim=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
        # Load features data
        with open(features_file, 'rb') as f:
            features_data = pickle.load(f)
        
        self.features = features_data['features']['fused']  # Using fused features (global + local)
        self.cell_ids = features_data['cell_ids']
        
        # Determine input and output dimensions if not provided
        if input_dim is None:
            input_dim = self.features.shape[1]
        
        if output_dim is None:
            # Default output dimension if not provided
            output_dim = 1000  # This will be updated after loading the normalization info
        
        # Initialize model
        self.model = CustomPytorchRegressor(input_dim, output_dim)
        self.model.load_model(model_path)
        
        # Initialize normalization info
        self.normalization_info = None
        self.scaler = None
        
    def load_normalization_info(self, normalization_info_path):
        """
        Load normalization information
        Args:
            normalization_info_path: Path to the normalization info file
        """
        with open(normalization_info_path, 'rb') as f:
            self.normalization_info = pickle.load(f)
        
        # Initialize scaler with loaded parameters
        self.scaler = StandardScaler()
        self.scaler.mean_ = self.normalization_info['x_mean']
        self.scaler.scale_ = self.normalization_info['x_std']
        
        # Update output dimension if needed
        if hasattr(self.normalization_info, 'y_min') and len(self.normalization_info['y_min']) > 0:
            if hasattr(self.model, 'model'):
                output_layer = self.model.model.expression_branch
                if output_layer.out_features != len(self.normalization_info['y_min']):
                    print(f"Updating output dimension from {output_layer.out_features} to {len(self.normalization_info['y_min'])}")
                    # Recreate output layers with correct dimensions
                    self.model.model.expression_branch = torch.nn.Linear(
                        output_layer.in_features,
                        len(self.normalization_info['y_min'])
                    ).to(self.device)
                    self.model.model.binary_branch = torch.nn.Linear(
                        output_layer.in_features,
                        len(self.normalization_info['y_min'])
                    ).to(self.device)
                    # Reload model weights
                    self.model.load_model(self.model_path)
    
    def preprocess_features(self):
        """
        Preprocess features using the loaded scaler
        Returns:
            Preprocessed features
        """
        if self.scaler is not None:
            return self.scaler.transform(self.features)
        else:
            # If no scaler is available, use raw features
            print("Warning: No normalization information loaded. Using raw features.")
            return self.features
    
    def predict(self):
        """
        Generate predictions using the loaded model
        Returns:
            Predictions array
        """
        # Preprocess features
        preprocessed_features = self.preprocess_features()
        
        # Create a simple DataLoader for prediction
        from torch.utils.data import TensorDataset, DataLoader
        
        # Create dataset and dataloader
        dataset = TensorDataset(torch.tensor(preprocessed_features, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        
        # Generate predictions
        self.model.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch[0].to(self.device)
                
                # Forward pass
                expression_pred, binary_pred = self.model.model(features)
                
                # Apply sigmoid to binary predictions and combine with expression predictions
                sigmoid = torch.nn.Sigmoid()
                binary_prob = sigmoid(binary_pred)
                final_pred = binary_prob * expression_pred
                
                all_predictions.append(final_pred.cpu().numpy())
        
        # Concatenate all predictions
        predictions = np.concatenate(all_predictions, axis=0)
        
        return predictions
    
    def denormalize_predictions(self, predictions):
        """
        Denormalize predictions using the loaded normalization information
        Args:
            predictions: Normalized predictions
        Returns:
            Denormalized predictions
        """
        if self.normalization_info is None:
            print("Warning: No normalization information loaded. Returning raw predictions.")
            return predictions
        
        # Ensure predictions and normalization info have the same number of genes
        if predictions.shape[1] != len(self.normalization_info['y_min']):
            print(f"Warning: Mismatch between prediction dimensions ({predictions.shape[1]}) and normalization info dimensions ({len(self.normalization_info['y_min'])})")
            return predictions
        
        # Denormalize each gene
        denormalized = np.copy(predictions)
        for i in range(predictions.shape[1]):
            # Apply denormalization only to non-zero predictions
            mask = denormalized[:, i] > 0
            if mask.any():
                denormalized[mask, i] = denormalized[mask, i] * self.normalization_info['y_range'][i] + self.normalization_info['y_min'][i]
        
        return denormalized
    
    def save_predictions_to_h5ad(self, predictions, gene_names, output_file, reference_h5ad=None):
        """
        Save predictions to h5ad file format
        Args:
            predictions: Predicted gene expression values
            gene_names: List of gene names
            output_file: Path to output h5ad file
            reference_h5ad: Optional reference h5ad file to copy metadata from
        """
        # Ensure the number of genes matches
        if predictions.shape[1] != len(gene_names):
            raise ValueError(f"Mismatch between prediction dimensions ({predictions.shape[1]}) and gene names ({len(gene_names)})")
        
        # Create AnnData object
        adata = ad.AnnData(
            X=predictions,
            obs=pd.DataFrame(index=self.cell_ids),
            var=pd.DataFrame(index=gene_names)
        )
        
        # Add metadata from reference h5ad if provided
        if reference_h5ad is not None:
            ref_adata = ad.read_h5ad(reference_h5ad)
            
            # Copy cell metadata if available
            for key in ref_adata.obs.columns:
                if key in adata.obs.index:
                    adata.obs[key] = ref_adata.obs.loc[adata.obs.index, key]
            
            # Copy gene metadata if available
            for key in ref_adata.var.columns:
                if key in adata.var.index:
                    adata.var[key] = ref_adata.var.loc[adata.var.index, key]
            
            # Copy unstructured metadata
            adata.uns = ref_adata.uns.copy()
        
        # Save to h5ad file
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        adata.write_h5ad(output_file)
        print(f"Predictions saved to {output_file}")

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Predict gene expression using trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--features_file', type=str, required=True, help='Path to features file (.pkl)')
    parser.add_argument('--normalization_info', type=str, default=None, help='Path to normalization info file (.pkl)')
    parser.add_argument('--output_file', type=str, required=True, help='Path to output h5ad file')
    parser.add_argument('--reference_h5ad', type=str, default=None, help='Optional reference h5ad file for metadata')
    parser.add_argument('--gene_list_file', type=str, default=None, help='Path to gene list file (.txt)')
    parser.add_argument('--input_dim', type=int, default=None, help='Input feature dimension (optional)')
    parser.add_argument('--output_dim', type=int, default=None, help='Output dimension (number of genes, optional)')
    return parser.parse_args()

def load_gene_names(gene_list_file=None, reference_h5ad=None):
    """
    Load gene names from file or reference h5ad
    Args:
        gene_list_file: Path to gene list file
        reference_h5ad: Path to reference h5ad file
    Returns:
        List of gene names
    """
    if gene_list_file is not None:
        with open(gene_list_file, 'r') as f:
            gene_names = [line.strip() for line in f if line.strip()]
        return gene_names
    elif reference_h5ad is not None:
        ref_adata = ad.read_h5ad(reference_h5ad)
        return list(ref_adata.var.index)
    else:
        # Default gene names if neither file is provided
        print("Warning: No gene list or reference h5ad provided. Using generic gene names.")
        return [f"gene_{i}" for i in range(1000)]  # Default to 1000 genes

def main():
    """
    Main function for gene expression prediction
    """
    args = parse_args()
    
    # Load gene names
    gene_names = load_gene_names(args.gene_list_file, args.reference_h5ad)
    
    # Initialize predictor
    print(f"Initializing predictor with model: {args.model_path}")
    predictor = Predictor(
        model_path=args.model_path,
        features_file=args.features_file,
        input_dim=args.input_dim,
        output_dim=args.output_dim or len(gene_names)
    )
    
    # Load normalization information if provided
    if args.normalization_info is not None:
        print(f"Loading normalization information from: {args.normalization_info}")
        predictor.load_normalization_info(args.normalization_info)
    
    # Generate predictions
    print("Generating predictions...")
    start_time = time.time()
    
    predictions = predictor.predict()
    
    # Denormalize predictions if normalization info is available
    if predictor.normalization_info is not None:
        predictions = predictor.denormalize_predictions(predictions)
    
    elapsed_time = time.time() - start_time
    print(f"Prediction completed in {elapsed_time:.2f} seconds")
    print(f"Prediction shape: {predictions.shape} - (number of cells, number of genes)")
    
    # Save predictions to h5ad file
    predictor.save_predictions_to_h5ad(
        predictions=predictions,
        gene_names=gene_names,
        output_file=args.output_file,
        reference_h5ad=args.reference_h5ad
    )

if __name__ == '__main__':
    main()