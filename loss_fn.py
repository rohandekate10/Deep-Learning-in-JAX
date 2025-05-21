import jax
import jax.numpy as jnp
import optax

def mse_loss(model, x, y):
    """Compute mean squared error loss over a batch."""
    def single_example_loss(model, x, y):
        return (model(x) - y) ** 2
    batch_loss = jax.vmap(single_example_loss, in_axes=(None, 0, 0))
    return batch_loss(model, x, y).mean()

def cross_entropy_loss(model, state, x, y):
    """Compute cross entropy loss over a batch."""
    # Create a batch of models
    batch_model = jax.vmap(
        model, axis_name="batch", in_axes=(0, None), out_axes=(0, None)
    )
    # Apply the batch of models to the batch of inputs
    logits, state = batch_model(x, state)
    
    # Convert labels to one-hot encoding
    num_classes = logits.shape[-1]
    y_one_hot = jax.nn.one_hot(y, num_classes)

    # Compute loss and accuracy
    loss = optax.softmax_cross_entropy(logits, y_one_hot).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    
    # Return loss as the main output and (accuracy, state) as auxiliary information
    return loss, (accuracy, state)