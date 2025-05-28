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
    create_data_loaders, train_step_cross_entropy, eval_step_cross_entropy, TensorBoardLogger,
    save_best_model, load_model, check_device_jax
)
from optimizer import create_optimizer
from models import MLPClassifier
import argparse
import os
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms

#--------------------------------
DATASET_PATH = './data/'

# Transformations applied on each image => bring them into a numpy array
DATA_MEANS = np.array([0.49139968, 0.48215841, 0.44653091])
DATA_STD = np.array([0.24703223, 0.24348513, 0.26158784])
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - DATA_MEANS) / DATA_STD
    return img

test_transform = image_to_numpy
# For training, we add some augmentation. Networks are too powerful and would overfit.
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      image_to_numpy])
    
def main(args):
    device = check_device_jax()
    print(f"Using device: {device}")
    # Create checkpoints directory if it doesn't exist
    os.makedirs('checkpoints', exist_ok=True)
    model_path = os.path.join('checkpoints', 'best_model.pkl')
    
    # Set up data
    train_set = CIFAR10(root=DATASET_PATH, train=True, transform=train_transform, download=True)
    val_set = CIFAR10(root=DATASET_PATH, train=True, transform=test_transform, download=True)
    train_set, _ = data.random_split(train_set, [45000, 5000], generator=torch.Generator().manual_seed(42))
    _, val_set = data.random_split(val_set, [45000, 5000], generator=torch.Generator().manual_seed(42))

    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_transform, download=True)

    # Create keys
    key_model, key_model_init = jax.random.split(jax.random.PRNGKey(args.seed), 2)

    train_loader, val_loader, test_loader = create_data_loaders(
        train_set, val_set, test_set,
        train=[True, False, False],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed+np.random.randint(0, 1000000)
    )

    # Initialize model and optimizer
    model, state = MLPClassifier(in_size=32*32*3, 
                                 out_size=10, 
                                 width_size=512, 
                                 depth=3, 
                                 activation=jax.nn.swish, 
                                 dropout_rate=0.4, 
                                 key=key_model_init)

    # Initialize optimizer
    optimizer = create_optimizer(
        optimizer_name='adam',
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        num_steps_per_epoch=len(train_loader),
        weight_decay=2e-4
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    
    # Initialize TensorBoard logger if enabled
    logger = TensorBoardLogger() if args.use_tensorboard else None
    
    # Training loop
    pbar = tqdm(range(args.num_epochs), desc="Training")
    best_val_accuracy = 0
    train_accuracies = []
    val_accuracies = []
    
    # Initialize random key for training
    key = jax.random.PRNGKey(args.seed+np.random.randint(0, 1000000))
    
    for epoch in pbar:
        # Training
        epoch_train_accuracy = []
        for x, y in train_loader:
            model, state, opt_state, accuracy = train_step_cross_entropy(
                model, opt_state, optimizer, x, y, state
            )
            epoch_train_accuracy.append(accuracy)
        
        # Validation
        inference_model = eqx.nn.inference_mode(model)
        inference_model = eqx.Partial(inference_model, state=state)

        epoch_val_accuracy = []
        for x, y in val_loader:
            pred = eval_step_cross_entropy(
                inference_model, x
            )
            accuracy = jnp.mean(jnp.argmax(pred, axis=-1) == y)
            epoch_val_accuracy.append(accuracy)
        
        # Calculate average losses
        avg_train_accuracy = np.mean(epoch_train_accuracy)
        avg_val_accuracy = np.mean(epoch_val_accuracy)
        
        # Update best validation accuracy
        if avg_val_accuracy > best_val_accuracy:
            best_val_accuracy = avg_val_accuracy
            # Save best model
            eqx.tree_serialise_leaves(model_path, model)
        
        # Save losses for plotting
        train_accuracies.append(avg_train_accuracy)
        val_accuracies.append(avg_val_accuracy)

        # Log metrics to TensorBoard if enabled
        if logger is not None:
            logger.log_metrics({
                'learning_rate': args.learning_rate,
                'train/accuracy': avg_train_accuracy,
                'val/accuracy': avg_val_accuracy
            }, epoch)
            
            # Log parameter histograms every 10 epochs
            if epoch % 10 == 0:
                params = eqx.filter(model, eqx.is_array)
                logger.log_parameters(params, epoch)
        
        # Update progress bar
        pbar.set_postfix(
            train_accuracy=f"{avg_train_accuracy:.2f}",
            val_accuracy=f"{avg_val_accuracy:.2f}",
            best_val_accuracy=f"{best_val_accuracy:.2f}"
        )
    
    # Close logger if enabled
    if logger is not None:
        logger.close()
    
    # Plot the loss curves
    plt.figure(figsize=(8, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.legend()
    plt.savefig('accuracy_curves.png')
    plt.close()
    
    # Load best model and evaluate on test set
    best_model = load_model(model_path, inference_model)
    test_preds = []
    test_labels = []
    for x, y in test_loader:
        pred = jax.vmap(eval_step_cross_entropy, in_axes=(None, 0))(best_model, x)
        test_preds.append(pred)
        test_labels.append(y)
    
    test_preds = jnp.concatenate(test_preds, axis=0)
    test_labels = jnp.concatenate(test_labels, axis=0)
    accuracy = jnp.mean(jnp.argmax(test_preds, axis=-1) == test_labels)
    print(f"Test accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--use_tensorboard", action="store_true", help="Enable TensorBoard logging")
    args = parser.parse_args()
    main(args)