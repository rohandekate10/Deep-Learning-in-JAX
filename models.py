import equinox as eqx
import jax
import jax.numpy as jnp

class Linear_MLP(eqx.Module):
    """A flexible Multi-Layer Perceptron implemented using Equinox.
    
    Args:
        in_size: Input dimension
        out_size: Output dimension
        hidden_sizes: List of hidden layer sizes
        activation: Activation function to use between layers
        key: PRNG key for initialization
    """
    layers: list
    activation: callable

    def __init__(self, in_size, out_size, hidden_sizes, activation=jax.nn.relu, *, key):
        keys = jax.random.split(key, len(hidden_sizes) + 1)
        
        # Build list of layers
        sizes = [in_size] + list(hidden_sizes) + [out_size]
        self.layers = []
        
        # Create linear layers
        for i in range(len(sizes) - 1):
            self.layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], use_bias=True, key=keys[i]))
            
        self.activation = activation

    def __call__(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, in_size)
            
        Returns:
            Output tensor of shape (batch_size, out_size)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Don't apply activation after the last layer
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x
    
class MLP(eqx.Module):
    """A flexible Multi-Layer Perceptron implemented using Equinox.
    
    Args:
        in_size: Input dimension
        out_size: Output dimension
        width_size: Size of hidden layers
        depth: Number of hidden layers
        activation: Activation function to use between layers
        final_activation: Activation function after the final layer
        use_bias: Whether to use bias in internal layers
        use_final_bias: Whether to use bias in final layer
        key: PRNG key for initialization
    """
    layers: list
    activation: callable
    final_activation: callable

    def __init__(self, in_size, out_size, width_size, depth, 
                activation=jax.nn.relu, final_activation=lambda x: x,
                use_bias=True, use_final_bias=True, *, key):
        keys = jax.random.split(key, depth + 1)
        
        # Build list of layers
        sizes = [in_size] + [width_size] * (depth - 1) + [out_size]
        self.layers = []
        
        # Create linear layers
        for i in range(len(sizes) - 1):
            use_layer_bias = use_bias if i < len(sizes) - 2 else use_final_bias
            self.layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], 
                                            use_bias=use_layer_bias, 
                                            key=keys[i]))
            
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, in_size)
            
        Returns:
            Output tensor of shape (batch_size, out_size)
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply activation after each layer except the last
            if i < len(self.layers) - 1:
                x = self.activation(x)
        # Apply final activation
        x = self.final_activation(x)
        return x
    
@eqx.nn.make_with_state
class MLPClassifier(eqx.Module):
    """A flexible Multi-Layer Perceptron implemented using Equinox.
    
    Args:
        in_size: Input dimension
        out_size: Output dimension
        width_size: Size of hidden layers
        depth: Number of hidden layers
        activation: Activation function to use between layers
        final_activation: Activation function after the final layer
        use_bias: Whether to use bias in internal layers
        use_final_bias: Whether to use bias in final layer
        key: PRNG key for initialization
    """
    layers: list
    activation: callable
    final_activation: callable
    dropout: eqx.nn.Dropout
    batch_norm: eqx.nn.BatchNorm

    def __init__(self, in_size=32*32*3, out_size=10, width_size=512, depth=3, 
                activation=jax.nn.relu, final_activation=lambda x: x,
                use_bias=True, use_final_bias=True, dropout_rate=0.25, *, key):
        keys = jax.random.split(key, depth + 1)
        
        # Build list of layers
        sizes = [in_size] + [width_size] * (depth - 1) + [out_size]
        self.layers = []
        
        # Create linear layers
        for i in range(len(sizes) - 1):
            use_layer_bias = use_bias if i < len(sizes) - 2 else use_final_bias
            self.layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], 
                                            use_bias=use_layer_bias, 
                                            key=keys[i]))
            
        self.dropout = eqx.nn.Dropout(dropout_rate)
        self.batch_norm = eqx.nn.BatchNorm(width_size, "batch")
        self.activation = activation
        self.final_activation = final_activation

    def __call__(self, x, state, *, key=None):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, height, width, channels)
            state: BatchNorm state
            key: PRNG key for initialization
            
        Returns:
            Output tensor of shape (batch_size, out_size)
            state: BatchNorm state
        """
        if key is None:
            key = jax.random.PRNGKey(0)
        key1, key2 = jax.random.split(key, 2)
        
        x = x.ravel() # Flatten the input
        # First layer
        x = self.layers[0](x)  # (batch_size, width_size)
        x = self.activation(x)
        
        # Hidden layers with batch norm
        for i, layer in enumerate(self.layers[1:-1], 1):
            x = self.dropout(x, key=key1)
            x = layer(x)
            x, state = self.batch_norm(x, state)
            x = self.activation(x)
        
        # Final layer
        x = self.dropout(x, key=key2)
        x = self.layers[-1](x)
        x = self.final_activation(x)
        
        return x, state

