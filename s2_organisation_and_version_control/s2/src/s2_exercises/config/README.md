# MNIST Training Configuration

This directory contains Hydra configuration files for reproducible MNIST model training.

## Configuration Structure

The configuration is organized into multiple layers:

### 1. Model Configurations (`model/`)
Controls the CNN architecture hyperparameters:
- `cnn_default.yaml`: Default model (32→64→128 channels)
- `cnn_small.yaml`: Smaller model (16→32→64 channels)
- `cnn_large.yaml`: Larger model (64→128→256 channels)

**Parameters:**
- `conv{1,2,3}_out_channels`: Number of output channels for each conv layer
- `conv{1,2,3}_kernel_size`: Kernel size for each conv layer
- `conv{1,2,3}_stride`: Stride for each conv layer
- `dropout_rate`: Dropout rate
- `fc1_out_features`: Output features (number of classes)

### 2. Training Configurations (`training/`)
Controls training hyperparameters and settings:
- `default.yaml`: Standard training (5 epochs, batch_size=32)
- `quick.yaml`: Fast training for testing (2 epochs, batch_size=64)
- `long.yaml`: Extended training (10 epochs)

**Parameters:**
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `seed`: Random seed for reproducibility
- `device`: Device selection ('auto', 'cuda', 'mps', 'cpu')
- `save_model`: Whether to save the trained model
- `model_path`: Path to save the model
- `save_plots`: Whether to save training plots
- `plot_path`: Path to save plots
- `log_interval`: Logging frequency (iterations)

### 3. Optimizer Configurations (`optimizer/`)
Controls optimizer settings:
- `adam.yaml`: Adam optimizer
- `nesterov.yaml`: SGD with Nesterov momentum

### 4. Experiment Configurations (`experiment/`)
Combines model, training, and optimizer settings for complete experiments:
- `exp1.yaml`: Small model + quick training + Adam
- `exp2.yaml`: Default model + default training + Adam
- `exp3.yaml`: Large model + long training + Nesterov

## Usage

### Run with default configuration:
```bash
uv run python src/s2_exercises/train.py
```

### Override individual components:
```bash
# Use small model
uv run python src/s2_exercises/train.py model=cnn_small

# Use quick training
uv run python src/s2_exercises/train.py training=quick

# Use Nesterov optimizer
uv run python src/s2_exercises/train.py optimizer=nesterov

# Combine multiple overrides
uv run python src/s2_exercises/train.py model=cnn_large training=long optimizer=nesterov
```

### Run predefined experiments:
```bash
# Experiment 1: Quick test
uv run python src/s2_exercises/train.py +experiment=exp1

# Experiment 2: Full training
uv run python src/s2_exercises/train.py +experiment=exp2

# Experiment 3: Large model training
uv run python src/s2_exercises/train.py +experiment=exp3
```

### Override specific parameters from command line:
```bash
# Change learning rate
uv run python src/s2_exercises/train.py optimizer.lr=0.001

# Change batch size
uv run python src/s2_exercises/train.py training.batch_size=64

# Change number of epochs
uv run python src/s2_exercises/train.py training.epochs=20

# Change dropout rate
uv run python src/s2_exercises/train.py model.dropout_rate=0.3
```

## Configuration Hierarchy

The configuration loading follows this hierarchy:
1. Base config (`config.yaml`)
2. Group configs (model, training, optimizer)
3. Experiment configs (if specified)
4. Command-line overrides (highest priority)

## Adding New Configurations

### Add a new model architecture:
1. Create `config/model/my_model.yaml`
2. Define all model parameters
3. Use with: `python train.py model=my_model`

### Add a new training configuration:
1. Create `config/training/my_training.yaml`
2. Define training parameters
3. Use with: `python train.py training=my_training`

### Add a new experiment:
1. Create `config/experiment/exp4.yaml`
2. Use `@package _global_` at the top
3. Define defaults with overrides
4. Use with: `python train.py +experiment=exp4`

## Reproducibility

All experiments are reproducible thanks to:
- Fixed random seeds (`training.seed`)
- Complete configuration logging
- Version-controlled config files
- Hydra's automatic output directory management
