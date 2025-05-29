import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
import torch
import torch.utils.data as data
import optax
from tqdm import tqdm
from trainer_module import (
    create_data_loaders, train_step, eval_step, TensorBoardLogger,
    save_best_model, load_model, check_device_jax, make_predictions
)
from models import Linear_MLP, MLP
import argparse
import os

def target_function(x):
    return np.sin(x * 3.0)

class RegressionDataset(data.Dataset):
    def __init__(self, num_points, seed):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.x = rng.uniform(low=-2.0, high=2.0, size=(num_points, 1))
        self.y = target_function(self.x)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
def main(args):
    device = check_device_jax()
    print(f"Using device: {device}")
    # Create checkpoints directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'best_model.pkl')
    
    # Set up data
    train_set = RegressionDataset(num_points=args.num_points, seed=args.seed)
    val_set = RegressionDataset(num_points=200, seed=args.seed+1)
    test_set = RegressionDataset(num_points=500, seed=args.seed+2)
    
    train_loader, val_loader, test_loader = create_data_loaders(
        train_set, val_set, test_set,
        train=[True, False, False],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed
    )

    # Plot data
    x = np.linspace(-2, 2, 1000)
    plt.figure(figsize=(8, 6))
    plt.scatter(train_set.x, train_set.y, color='C1', marker='x', alpha=0.5, label='Training set')
    plt.plot(x, target_function(x), linewidth=3.0, label='Ground Truth Function')
    plt.legend()
    plt.title('Regression function')
    plt.savefig(os.path.join(args.output_dir, 'regression_function.png'))
    plt.close()

    # Initialize model and optimizer
    model = MLP(
        in_size=1,
        out_size=1,
        width_size=128,
        depth=2,
        activation=jax.nn.relu,
        key=jax.random.PRNGKey(args.seed)
    )

    optimizer = optax.adam(learning_rate=args.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    # Initialize TensorBoard logger if enabled
    logger = TensorBoardLogger() if args.use_tensorboard else None
    
    # Training loop
    pbar = tqdm(range(args.num_epochs), desc="Training")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in pbar:
        # Training
        epoch_train_losses = []
        for x, y in train_loader:
            model, opt_state, loss = train_step(model, opt_state, optimizer, x, y)
            epoch_train_losses.append(loss)
    
        # Validation
        epoch_val_losses = []
        for x, y in val_loader:
            val_loss = eval_step(model, x, y)
            epoch_val_losses.append(val_loss)
        
        # Calculate average losses
        avg_train_loss = np.mean(epoch_train_losses)
        avg_val_loss = np.mean(epoch_val_losses)
        
        # Save losses for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        # Save best model
        best_val_loss = save_best_model(model, avg_val_loss, best_val_loss, model_path)
        
        # Log metrics to TensorBoard if enabled
        if logger is not None:
            logger.log_metrics({
                'train/loss': avg_train_loss,
                'val/loss': avg_val_loss,
                'learning_rate': args.learning_rate
            }, epoch)
            
            # Log parameter histograms every 10 epochs
            if epoch % 10 == 0:
                params = eqx.filter(model, eqx.is_array)
                logger.log_parameters(params, epoch)
        
        # Update progress bar
        pbar.set_postfix(
            train_loss=f"{avg_train_loss:.4f}",
            val_loss=f"{avg_val_loss:.4f}",
            best_val_loss=f"{best_val_loss:.4f}"
        )
    
    # Close logger if enabled
    if logger is not None:
        logger.close()
    
    # Plot the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(args.output_dir, 'loss_curves.png'))
    plt.close()
    
    # Load best model and evaluate on test set
    best_model = load_model(model_path, model)
    test_loss = eval_step(best_model, test_loader.dataset.x, test_loader.dataset.y)
    print(f"Best model test loss: {test_loss:.4f}")

    # Model predictions
    x_test = jnp.array(test_loader.dataset.x)  # Convert to JAX array
    predictions = make_predictions(best_model, x_test)

    # Plot the predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(test_set.x, test_set.y, color='C1', marker='x', alpha=0.5, label='Test set')
    plt.scatter(x_test, predictions, color='C0', marker='.', alpha=0.5, label='Predictions')
    x_plot = np.linspace(-2, 2, 1000)
    plt.plot(x_plot, target_function(x_plot), linewidth=3.0, label='Ground Truth Function')
    plt.legend()
    plt.title('Regression Results')
    plt.savefig(os.path.join(args.output_dir, 'regression_results.png'))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="output/regression")
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    args = parser.parse_args()
    main(args)