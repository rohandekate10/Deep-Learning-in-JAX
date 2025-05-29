# Standard libraries
import os
import sys
from typing import Any, Sequence, Optional, Tuple, Iterator, Dict, Callable, Union
import json
import time
from tqdm.auto import tqdm
import numpy as np
from copy import copy
from glob import glob
from collections import defaultdict
import pickle
from datetime import datetime

# JAX/Equinox
import jax
import jax.numpy as jnp
from jax import random
import equinox as eqx
import optax

# PyTorch for data loading
import torch
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

# Local imports
from loss_fn import mse_loss, cross_entropy_loss

def check_device_torch():
    if torch.cuda.is_available():
        print("CUDA is available")
        return torch.device("cuda")
    else:
        print("CUDA is not available")
        return torch.device("cpu")

def check_device_jax():
    return jax.devices()

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def create_data_loaders(*datasets, train=True, batch_size=128, num_workers=4, seed=42):
    """Creates data loaders for training and evaluation."""
    if not isinstance(train, (list, tuple)):
        train = [train for _ in datasets]
    
    loaders = []
    for dataset, is_train in zip(datasets, train):
        loader = data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            drop_last=is_train,
            collate_fn=numpy_collate,
            num_workers=num_workers,
            persistent_workers=is_train,
            generator=torch.Generator().manual_seed(seed)
        )
        loaders.append(loader)
    return loaders

@eqx.filter_jit
def train_step(model, opt_state, optimizer, x, y):
    """Single training step."""
    def loss_fn(params):
        model_with_params = eqx.apply_updates(model, params)
        return mse_loss(model_with_params, x, y)
    
    params = eqx.filter(model, eqx.is_array)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss

@eqx.filter_jit
def eval_step(model, x, y):
    """Single evaluation step."""
    return mse_loss(model, x, y)

@eqx.filter_jit
def make_predictions(model, x):
    """Make predictions for a given model and input."""
    def single_example_prediction(model, x):
        return model(x)
    batch_prediction = jax.vmap(single_example_prediction, in_axes=(None, 0))
    return batch_prediction(model, x)

@eqx.filter_jit
def train_step_cross_entropy(model, opt_state, optimizer, x, y, state):
    """Single training step for cross entropy loss."""
    loss, (accuracy, state) = eqx.filter_grad(cross_entropy_loss, has_aux=True)(model, state, x, y)
    updates, opt_state = optimizer.update(loss, opt_state)
    model = eqx.apply_updates(model, updates)
    return model, state, opt_state, accuracy

@eqx.filter_jit
def eval_step_cross_entropy(model, x):
    pred, _ = jax.vmap(model)(x)
    return pred

def save_model(model, path):
    """Save model using Equinox's serialization."""
    eqx.tree_serialise_leaves(path, model)

def load_model(path, model_template):
    """Load model using Equinox's serialization."""
    return eqx.tree_deserialise_leaves(path, model_template)

def save_best_model(model, val_loss, best_val_loss, save_path):
    """Save model if it has the best validation loss."""
    if val_loss < best_val_loss:
        save_model(model, save_path)
        return val_loss
    return best_val_loss

class TensorBoardLogger:
    def __init__(self, log_dir="runs"):
        """Initialize TensorBoard logger."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = os.path.join(log_dir, timestamp)
        self.writer = SummaryWriter(self.log_dir)
        
    def log_metrics(self, metrics, step):
        """Log metrics to TensorBoard."""
        for name, value in metrics.items():
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                value = float(value)
            self.writer.add_scalar(name, value, step)
            
    def log_histogram(self, name, values, step):
        """Log histogram of values to TensorBoard."""
        if isinstance(values, jnp.ndarray):
            values = np.array(values)
        self.writer.add_histogram(name, values, step)
        
    def log_parameters(self, params, step):
        """Log parameter histograms from a PyTree."""
        def _log_param(path, value):
            if isinstance(value, (jnp.ndarray, np.ndarray)):
                name = "/".join(str(p) for p in path)
                self.log_histogram(f"parameters/{name}", value, step)
        
        jax.tree_util.tree_map_with_path(_log_param, params)
        
    def close(self):
        """Close the TensorBoard writer."""
        self.writer.close()

