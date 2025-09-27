import os
import time
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import anndata as ad
from sklearn.metrics import r2_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Global parameters based on technical description
hparams = {
    'num_hidden_layers': 4,
    'hidden_dims': [512, 512, 1024, 1024],
    'dropout_rate': 0.1,
    'learning_rate': 0.01,
    'weight_decay': 0.05,
    'batch_size': 256,
    'num_epochs': 100,
    'patience': 10,
    'loss_weights': {
        'lambda1': 1.0,
        'lambda2': 5.0,
        'alpha': 1.0,
        'gamma': 20.0
    }
}

class GeneExpressionDataset(Dataset):
    """
    Dataset class for gene expression prediction
    Args:
        features: Input features (histology features)
        labels: Target gene expression values
    """
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

class FocalLoss(nn.Module):
    """
    Focal Loss implementation to handle class imbalance
    Args:
        alpha: Weighting factor for positive class
        gamma: Focusing parameter to down-weight easy examples
    """
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, inputs, targets):
        # Compute binary cross-entropy loss
        bce_loss = self.bce_loss(inputs, targets)
        
        # Compute probabilities and focal loss weights
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()

class DualOutputMLP(nn.Module):
    """
    Dual-output neural network for gene expression prediction
    Args:
        input_dim: Dimension of input features
        output_dim: Number of genes to predict
    """
    def __init__(self, input_dim, output_dim):
        super(DualOutputMLP, self).__init__()
        
        # Create hidden layers
        layers = []
        current_dim = input_dim
        
        # Add hidden layers as specified in hparams
        for hidden_dim in hparams['hidden_dims']:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(negative_slope=0.1))
            layers.append(nn.Dropout(p=hparams['dropout_rate']))
            current_dim = hidden_dim
        
        # Create feature extractor
        self.feature_extractor = nn.Sequential(*layers)
        
        # Create output branches
        self.expression_branch = nn.Linear(current_dim, output_dim)
        self.binary_branch = nn.Linear(current_dim, output_dim)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        expression_output = self.expression_branch(features)
        binary_output = self.binary_branch(features)
        
        return expression_output, binary_output

class CustomPytorchRegressor:
    """
    Custom regressor for gene expression prediction
    Args:
        input_dim: Dimension of input features
        output_dim: Number of genes to predict
    """
    def __init__(self, input_dim, output_dim):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = DualOutputMLP(input_dim, output_dim).to(self.device)
        
        # Initialize loss functions
        self.binary_loss_fn = FocalLoss(alpha=hparams['loss_weights']['alpha'],
                                        gamma=hparams['loss_weights']['gamma'])
        self.regression_loss_fn = nn.MSELoss()
        
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=hparams['learning_rate'],
            weight_decay=hparams['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Early stopping parameters
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_model = None
        
    def compute_class_weights(self, y_true):
        """
        Compute class weights to handle imbalance
        Args:
            y_true: Ground truth expression values
        Returns:
            Class weights tensor
        """
        # Determine binary labels (gene expressed or not)
        binary_labels = (y_true > 0).astype(int)
        
        # Count positive and negative samples
        pos_count = np.sum(binary_labels)
        neg_count = len(binary_labels) - pos_count
        
        # Calculate class weights
        if pos_count > 0 and neg_count > 0:
            pos_weight = neg_count / pos_count
            neg_weight = pos_count / neg_count
            
            # Create weight array
            weights = np.where(binary_labels == 1, pos_weight, neg_weight)
        else:
            weights = np.ones_like(y_true)
        
        return torch.tensor(weights, dtype=torch.float32).to(self.device)
    
    def compute_masked_mse(self, y_pred, y_true, mask):
        """
        Compute MSE loss only on non-zero values
        Args:
            y_pred: Predicted values
            y_true: True values
            mask: Mask indicating non-zero positions
        Returns:
            Masked MSE loss
        """
        if mask.sum() > 0:
            masked_pred = y_pred[mask]
            masked_true = y_true[mask]
            return self.regression_loss_fn(masked_pred, masked_true)
        return torch.tensor(0.0, device=self.device)
    
    def compute_loss(self, expression_pred, binary_pred, y_true):
        """
        Compute combined loss function
        Args:
            expression_pred: Predicted expression values
            binary_pred: Predicted binary labels
            y_true: True expression values
        Returns:
            Total loss
        """
        # Create binary labels (gene expressed or not)
        binary_true = (y_true > 0).float()
        
        # Compute class weights
        class_weights = self.compute_class_weights(y_true.cpu().detach().numpy())
        
        # Compute classification loss with weights
        binary_loss = self.binary_loss_fn(binary_pred, binary_true)
        weighted_binary_loss = binary_loss * class_weights.mean()
        
        # Compute regression loss only on non-zero values
        mask = y_true > 0
        regression_loss = self.compute_masked_mse(expression_pred, y_true, mask)
        
        # Combine losses with weights from hparams
        total_loss = (hparams['loss_weights']['lambda1'] * weighted_binary_loss + 
                     hparams['loss_weights']['lambda2'] * regression_loss)
        
        return total_loss
    
    def train_epoch(self, train_loader):
        """
        Train model for one epoch
        Args:
            train_loader: DataLoader for training data
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            features = batch['features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            expression_pred, binary_pred = self.model(features)
            
            # Compute loss
            loss = self.compute_loss(expression_pred, binary_pred, labels)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * features.size(0)
        
        return total_loss / len(train_loader.dataset)
    
    def evaluate(self, val_loader):
        """
        Evaluate model on validation data
        Args:
            val_loader: DataLoader for validation data
        Returns:
            Average validation loss, predictions, and true values
        """
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_true = []
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                expression_pred, binary_pred = self.model(features)
                
                # Apply sigmoid to binary predictions and combine with expression predictions
                sigmoid = nn.Sigmoid()
                binary_prob = sigmoid(binary_pred)
                final_pred = binary_prob * expression_pred
                
                # Compute loss
                loss = self.compute_loss(expression_pred, binary_pred, labels)
                
                total_loss += loss.item() * features.size(0)
                all_predictions.append(final_pred.cpu().numpy())
                all_true.append(labels.cpu().numpy())
        
        # Concatenate all predictions and true values
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_true = np.concatenate(all_true, axis=0)
        
        return total_loss / len(val_loader.dataset), all_predictions, all_true
    
    def fit(self, train_loader, val_loader):
        """
        Train model with early stopping
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        for epoch in range(hparams['num_epochs']):
            start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate on validation set
            val_loss, _, _ = self.evaluate(val_loader)
            
            # Update learning rate scheduler
            self.scheduler.step(val_loss)
            
            # Early stopping check
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model = self.model.state_dict()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= hparams['patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Print epoch statistics
            elapsed_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{hparams['num_epochs']}, "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Time: {elapsed_time:.2f}s")
        
        # Load best model weights
        if self.best_model is not None:
            self.model.load_state_dict(self.best_model)
    
    def predict(self, test_loader):
        """
        Make predictions on test data
        Args:
            test_loader: DataLoader for test data
        Returns:
            Predictions
        """
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                
                # Forward pass
                expression_pred, binary_pred = self.model(features)
                
                # Apply sigmoid to binary predictions and combine with expression predictions
                sigmoid = nn.Sigmoid()
                binary_prob = sigmoid(binary_pred)
                final_pred = binary_prob * expression_pred
                
                all_predictions.append(final_pred.cpu().numpy())
        
        # Concatenate all predictions
        all_predictions = np.concatenate(all_predictions, axis=0)
        
        return all_predictions
    
    def save_model(self, filepath):
        """
        Save model weights
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
    
    def load_model(self, filepath):
        """
        Load model weights
        Args:
            filepath: Path to load the model from
        """
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))

# Additional utility functions

def calculate_rmse(y_true, y_pred):
    """
    Calculate Root Mean Squared Error
    Args:
        y_true: True values
        y_pred: Predicted values
    Returns:
        RMSE value
    """
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def calculate_pcc(y_true, y_pred):
    """
    Calculate Pearson Correlation Coefficient
    Args:
        y_true: True values
        y_pred: Predicted values
    Returns:
        PCC value
    """
    # Compute for each gene
    n_genes = y_true.shape[1]
    pcc_scores = []
    
    for i in range(n_genes):
        true_vector = y_true[:, i]
        pred_vector = y_pred[:, i]
        
        # Check if all values are the same
        if np.std(true_vector) == 0 or np.std(pred_vector) == 0:
            pcc_scores.append(0.0)
        else:
            pcc_scores.append(np.corrcoef(true_vector, pred_vector)[0, 1])
    
    return np.mean(pcc_scores)

def parse_args():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(description='Train gene expression prediction model')
    parser.add_argument('--features_file', type=str, required=True, help='Path to features file (.pkl)')
    parser.add_argument('--expression_file', type=str, required=True, help='Path to gene expression file (.h5ad)')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--model_name', type=str, default='dual_output_model.pth', help='Name of the saved model')
    return parser.parse_args()

def main():
    """
    Main function for model training
    """
    args = parse_args()
    
    # Load features
    print(f"Loading features from {args.features_file}...")
    with open(args.features_file, 'rb') as f:
        features_data = pickle.load(f)
    
    features = features_data['features']['fused']  # Using fused features (global + local)
    cell_ids = features_data['cell_ids']
    
    # Load gene expression data
    print(f"Loading gene expression data from {args.expression_file}...")
    adata = ad.read_h5ad(args.expression_file)
    
    # Ensure cell IDs match and sort them
    common_cell_ids = list(set(cell_ids) & set(adata.obs.index))
    
    # Create index mappings
    feature_idx = [cell_ids.index(cell_id) for cell_id in common_cell_ids]
    adata_idx = [adata.obs.index.get_loc(cell_id) for cell_id in common_cell_ids]
    
    # Filter and reorder data
    filtered_features = features[feature_idx]
    filtered_expression = adata.X[adata_idx].toarray()
    
    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(filtered_features)
    
    # Split data into train, validation, and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        normalized_features, filtered_expression, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42  # 0.25 x 0.8 = 0.2 of total data
    )
    
    # Create datasets and dataloaders
    train_dataset = GeneExpressionDataset(X_train, y_train)
    val_dataset = GeneExpressionDataset(X_val, y_val)
    test_dataset = GeneExpressionDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=hparams['batch_size'], shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    input_dim = normalized_features.shape[1]
    output_dim = filtered_expression.shape[1]
    model = CustomPytorchRegressor(input_dim, output_dim)
    
    # Train model
    print("Starting model training...")
    model.fit(train_loader, val_loader)
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_loss, test_pred, test_true = model.evaluate(test_loader)
    
    # Calculate evaluation metrics
    rmse = calculate_rmse(test_true, test_pred)
    pcc = calculate_pcc(test_true, test_pred)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"PCC: {pcc:.4f}")
    
    # Save model
    model_path = os.path.join(args.output_dir, args.model_name)
    model.save_model(model_path)
    print(f"Model saved to {model_path}")
    
    # Save evaluation metrics
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"PCC: {pcc:.4f}\n")
    
    print(f"Evaluation metrics saved to {metrics_path}")

if __name__ == '__main__':
    main()
