import jax
import jax.numpy as jnp
import optax

def create_optimizer(optimizer_name:str, learning_rate:float, weight_decay:float, warmup:int=0, num_epochs:int=100, num_steps_per_epoch:int=1000):
    """
    Initializes the optimizer and learning rate scheduler.

    Args:
        optimizer_name: The name of the optimizer to use.
        learning_rate: The learning rate for the optimizer.
        weight_decay: The weight decay for the optimizer.
        warmup: The number of warmup steps for the learning rate scheduler.
        num_epochs: The number of epochs the model will be trained for.
        num_steps_per_epoch: The number of training steps per epoch.
    """

    # Initialize optimizer
    if optimizer_name.lower() == 'adam':
        opt_class = optax.adam
    elif optimizer_name.lower() == 'adamw':
        opt_class = optax.adamw
    elif optimizer_name.lower() == 'sgd':
        opt_class = optax.sgd
    else:
        assert False, f'Unknown optimizer "{optimizer_name}"'
    # Initialize learning rate scheduler
    # A cosine decay scheduler is used, but others are also possible
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup,
        decay_steps=int(num_epochs * num_steps_per_epoch),
        end_value=0.01 * learning_rate
    )
    # Clip gradients at max value, and evt. apply weight decay
    transf = [optax.clip_by_global_norm(1.0)]
    if opt_class == optax.sgd:  # wd is integrated in adamw
        transf.append(optax.add_decayed_weights(weight_decay))
    optimizer = optax.chain(
        *transf,
        opt_class(lr_schedule)
    )
    
    return optimizer