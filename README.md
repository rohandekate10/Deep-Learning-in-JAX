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
conda create -n jax-dl python=3.13
conda activate jax-dl
```

2. Install dependencies:
```bash
pip install jax jaxlib equinox optax torch torchvision tqdm matplotlib tensorboard
```

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

## Acknowledgments

This project is based on the UvA Deep Learning Tutorials by Phillip Lippe.

## References

@misc{lippe2024uvadlc,
   title        = {{UvA Deep Learning Tutorials}},
   author       = {Phillip Lippe},
   year         = 2024,
   howpublished = {\url{https://uvadlc-notebooks.readthedocs.io/en/latest/}}
}

@software{jax2018github,
  author = {James Bradbury and Roy Frostig and Peter Hawkins and Matthew James Johnson and Chris Leary and Dougal Maclaurin and George Necula and Adam Paszke and Jake Vander{P}las and Skye Wanderman-{M}ilne and Qiao Zhang},
  title = {{JAX}: composable transformations of {P}ython+{N}um{P}y programs},
  url = {http://github.com/jax-ml/jax},
  version = {0.3.13},
  year = {2018},
}

@article{kidger2021equinox,
    author={Patrick Kidger and Cristian Garcia},
    title={{E}quinox: neural networks in {JAX} via callable {P}y{T}rees and filtered transformations},
    year={2021},
    journal={Differentiable Programming workshop at Neural Information Processing Systems 2021}
}

@software{deepmind2020jax,
  title = {The {D}eep{M}ind {JAX} {E}cosystem},
  author = {DeepMind and Babuschkin, Igor and Baumli, Kate and Bell, Alison and Bhupatiraju, Surya and Bruce, Jake and Buchlovsky, Peter and Budden, David and Cai, Trevor and Clark, Aidan and Danihelka, Ivo and Dedieu, Antoine and Fantacci, Claudio and Godwin, Jonathan and Jones, Chris and Hemsley, Ross and Hennigan, Tom and Hessel, Matteo and Hou, Shaobo and Kapturowski, Steven and Keck, Thomas and Kemaev, Iurii and King, Michael and Kunesch, Markus and Martens, Lena and Merzic, Hamza and Mikulik, Vladimir and Norman, Tamara and Papamakarios, George and Quan, John and Ring, Roman and Ruiz, Francisco and Sanchez, Alvaro and Sartran, Laurent and Schneider, Rosalia and Sezener, Eren and Spencer, Stephen and Srinivasan, Srivatsan and Stanojevi\'{c}, Milo\v{s} and Stokowiec, Wojciech and Wang, Luyu and Zhou, Guangyao and Viola, Fabio},
  url = {http://github.com/google-deepmind},
  year = {2020},
}

@inbook{10.5555/3454287.3455008,
author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and K\"{o}pf, Andreas and Yang, Edward and DeVito, Zach and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
title = {PyTorch: an imperative style, high-performance deep learning library},
year = {2019},
publisher = {Curran Associates Inc.},
address = {Red Hook, NY, USA},
booktitle = {Proceedings of the 33rd International Conference on Neural Information Processing Systems},
articleno = {721},
numpages = {12}
}

@software{optax2020github,
  author = {DeepMind},
  title = {Optax: composable gradient transformation and optimisation, in JAX!},
  url = {https://github.com/deepmind/optax},
  year = {2020}
}

@software{matplotlib2003github,
  author = {Hunter, John D.},
  title = {Matplotlib: A 2D graphics environment},
  journal = {Computing in Science \& Engineering},
  volume = {9},
  number = {3},
  pages = {90--95},
  year = {2007},
  publisher = {IEEE Computer Society}
}

@software{tqdm2016github,
  author = {Casper da Costa-Luis},
  title = {tqdm: A Fast, Extensible Progress Bar for Python and CLI},
  url = {https://github.com/tqdm/tqdm},
  year = {2016}
}

@software{tensorboard2015github,
  author = {Google},
  title = {TensorBoard: TensorFlow's Visualization Toolkit},
  url = {https://github.com/tensorflow/tensorboard},
  year = {2015}
}