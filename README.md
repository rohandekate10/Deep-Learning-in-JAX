# JAX-based Deep Learning Framework

This repository contains implementations of deep learning models and training pipelines using JAX and Equinox. The framework is designed for two purposes:
1. Learning modern deep learning techniques using JAX's accelerated computing framework
2. Providing a foundation for research and experimentation

## Features

- **Model Architecture**
  - Flexible MLP with configurable depth and width
  - Batch normalization for training stability
  - Dropout for regularization
  - Swish activation function
  - Support for both classification and regression tasks

- **Training Pipeline**
  - PyTorch data loading with multi-worker support
  - Data augmentation (random horizontal flips)
  - Learning rate scheduling with Adam optimizer
  - Model checkpointing (best validation accuracy)
  - TensorBoard integration
  - Progress tracking with tqdm

## Setup

1. Create and activate conda environment:
```bash
conda create -n jax-dl python=3.12
conda activate jax-dl
```

2. Install dependencies:
```bash
pip install jax jaxlib equinox optax torch torchvision tqdm matplotlib tensorboard
```
If you're using an NVIDIA GPU install relevant packages for `jax` and `torch`.

3. Create data directory:
```bash
mkdir -p ./data
```

## Usage

### CIFAR-10 Classification

Train a model on CIFAR-10:
```bash
python CIFAR10_Classification.py --batch_size 256 --learning_rate 1e-3 --num_epochs 100 --use_tensorboard
```

Command-line arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--seed` | Random seed | 42 |
| `--batch_size` | Batch size | 256 |
| `--num_workers` | Data loading workers | 4 |
| `--learning_rate` | Learning rate | 1e-3 |
| `--num_epochs` | Training epochs | 100 |
| `--use_tensorboard` | Enable TensorBoard | False |

### Regression

Train a model on regression tasks:
```bash
python regression.py
```

Command-line arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--seed` | Random seed | 42 |
| `--batch_size` | Batch size | 64 |
| `--num_workers` | Data loading workers | 4 |
| `--learning_rate` | Learning rate | 1e-3 |
| `--num_epochs` | Training epochs | 10 |
| `--use_tensorboard` | Enable TensorBoard | False |
| `--param_log_freq` | Frequency for logging parameter histograms | 100 |
| `--plot_log_freq` | Frequency for logging prediction plots | 250 |
| `--width_size` | Width size of the MLP | 128 |
| `--depth` | Depth of the MLP | 2 |
| `--output_dir` | Directory for saving outputs | "output/regression" |
| `--num_points` | Number of training points | 1000 |

Example with custom parameters:
```bash
python regression.py --batch_size 128 --learning_rate 5e-4 --num_epochs 500 \
    --use_tensorboard --param_log_freq 50 --plot_log_freq 100 \
    --width_size 256 --depth 3
```

## Project Structure

```
.
├── CIFAR10_Classification.py  # Main classification script
├── models.py                  # Model architectures
├── trainer_module.py         # Training utilities
├── loss_fn.py               # Loss functions
├── optimizer.py             # Optimizer configurations
├── regression.py            # Regression example
├── checkpoints/             # Model checkpoints
└── runs/                    # TensorBoard logs
```

## Neural Network Architectures

| Architecture Type | Origin | Learning Type | Key Characteristics | Problem/Applications | Tasks & Subtasks | Advantages | Limitations/Drawbacks | Current Research Directions |
|------------------|---------|--------------|-------------------|---------------------|-----------------|------------|---------------------|---------------------------|
| Multilayer Perceptron (MLP) | F. Rosenblatt (1957) | Supervised | Feedforward, fully connected layers | Tabular data classification/regression | Tabular: Competitive for curated data | Simple, general non-linear function approximator | Not suited for high-dimensional data | As building blocks; efficient training |
| Convolutional Neural Network (CNN) | LeNet-5 (1998), AlexNet (2012) | Supervised, Self-supervised | Convolutional layers, pooling layers | Computer Vision | Image Classification, Object Detection | Excellent for spatial data | Struggles with long-range dependencies | Combining with attention/Transformers |
| Recurrent Neural Network (RNN) | David Rumelhart (1986) | Supervised, RL | Recurrent connections, hidden state | Sequential data | Machine Translation, Speech Recognition | Handles variable-length sequences | Vanishing/exploding gradients | Specialized use cases |
| Long Short-Term Memory (LSTM) | Hochreiter & Schmidhuber (1997) | Supervised, RL | Gated memory cells | Sequential data | Improved RNN tasks | Mitigates gradient issues | Computationally expensive | Continued use in niche areas |
| Gated Recurrent Unit (GRU) | Cho et al. (2014) | Supervised, RL | Simplified LSTM | Sequential data | Similar to LSTMs | Simpler than LSTMs | Similar to LSTMs | Less complex alternative to LSTMs |
| Transformer Network | Vaswani et al. (2017) | Self-supervised, Supervised | Self-attention mechanism | NLP, Computer Vision | Machine Translation, Text Generation | Captures long-range dependencies | Quadratic complexity | Efficiency improvements |
| Generative Adversarial Network (GAN) | Goodfellow et al. (2014) | Unsupervised | Generator and Discriminator | Image/video generation | Realistic Image Synthesis | Generates realistic data | Training instability | Improving stability |
| Variational Autoencoder (VAE) | Kingma & Welling (2013) | Unsupervised | Probabilistic latent space | Dimensionality reduction | Anomaly detection | Probabilistic framework | Blurry samples | More expressive decoders |
| Deep Q-Network (DQN) | DeepMind (2013-2015) | Reinforcement Learning | Q-value function | Game playing | Atari Games | Learns from raw inputs | Sample inefficient | Improved exploration |
| Actor-Critic (PPO, SAC) | Various (1990s-2018) | Reinforcement Learning | Actor and Critic networks | Robotics, control | Robotic Manipulation | Handles continuous actions | Sample inefficiency | Sample efficiency |
| Graph Neural Network (GNN) | Scarselli et al. (2008) | Supervised, Unsupervised | Message passing | Social networks, drug discovery | Node/Graph Classification | Handles graph data | Scalability issues | Scalability improvements |
| Physics-Informed Neural Network (PINN) | Raissi et al. (2017) | Supervised | Physics laws in loss | Differential equations | Solving PDEs | Leverages physics knowledge | Computationally expensive | Robustness & convergence |
| U-Net | Ronneberger et al. (2015) | Supervised | Encoder-decoder with skip connections | Medical Image Segmentation | Medical Image Segmentation | Excellent for segmentation | Specific to segmentation | 3D U-Nets, attention |

## Monitoring

Training progress can be monitored through:
- TensorBoard logs (if enabled)
- Accuracy curves (`accuracy_curves.png`)
- Best model checkpoints (`checkpoints/best_model.pkl`)

### Using TensorBoard

1. Enable TensorBoard logging during training by adding the `--use_tensorboard` flag:
```bash
python regression.py --use_tensorboard
```

2. Start TensorBoard to view the logs:
```bash
tensorboard --logdir=runs --port=6006
```

3. Open your web browser and navigate to:
```
http://localhost:6006
```

The TensorBoard interface provides several useful visualizations:
- **SCALARS**: View metrics like loss and learning rate over time
- **HISTOGRAMS**: Monitor parameter distributions (logged every `param_log_freq` epochs)
- **IMAGES**: View prediction plots (logged every `plot_log_freq` epochs)

You can control the logging frequency using:
- `--param_log_freq`: How often to log parameter histograms (default: 100 epochs)
- `--plot_log_freq`: How often to log prediction plots (default: 250 epochs)

To stop TensorBoard, press Ctrl+C in the terminal or close the terminal window.

## Acknowledgments

This project is based on the UvA Deep Learning Tutorials by Phillip Lippe.

## References

1. Lippe, P. (2024). UvA Deep Learning Tutorials. [https://uvadlc-notebooks.readthedocs.io/en/latest/](https://uvadlc-notebooks.readthedocs.io/en/latest/)

2. Bradbury, J., et al. (2018). JAX: Composable Transformations of Python+NumPy Programs. [http://github.com/jax-ml/jax](http://github.com/jax-ml/jax)

3. Kidger, P., & Garcia, C. (2021). Equinox: Neural Networks in JAX via Callable PyTrees and Filtered Transformations. Differentiable Programming workshop at Neural Information Processing Systems 2021.

4. DeepMind (2020). The DeepMind JAX Ecosystem. [http://github.com/google-deepmind](http://github.com/google-deepmind)

5. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. Proceedings of the 33rd International Conference on Neural Information Processing Systems.

6. DeepMind (2020). Optax: Composable Gradient Transformation and Optimisation, in JAX! [https://github.com/deepmind/optax](https://github.com/deepmind/optax)

7. Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering, 9(3), 90-95.

8. da Costa-Luis, C. (2016). tqdm: A Fast, Extensible Progress Bar for Python and CLI. [https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm)

9. Google (2015). TensorBoard: TensorFlow's Visualization Toolkit. [https://github.com/tensorflow/tensorboard](https://github.com/tensorflow/tensorboard)