import matplotlib.pyplot as plt
import torch
from data import corrupt_mnist
from model import Model
import hydra
import logging
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def get_device(device_config: str) -> torch.device:
    """Get the appropriate device based on configuration."""
    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_config)


@hydra.main(version_base=None, config_path="config", config_name="config")
def train(config: DictConfig) -> None:
    """Train a model on MNIST."""
    log.info("Training day and night")
    log.info(f"Configuration: \n{OmegaConf.to_yaml(config)}")

    # Set random seed for reproducibility
    if hasattr(config.training, 'seed'):
        torch.manual_seed(config.training.seed)

    # Get device
    DEVICE = get_device(config.training.device)
    log.info(f"Using device: {DEVICE}")

    # Initialize model with configuration
    model = Model(
        conv1_out_channels=config.model.conv1_out_channels,
        conv1_kernel_size=config.model.conv1_kernel_size,
        conv1_stride=config.model.conv1_stride,
        conv2_out_channels=config.model.conv2_out_channels,
        conv2_kernel_size=config.model.conv2_kernel_size,
        conv2_stride=config.model.conv2_stride,
        conv3_out_channels=config.model.conv3_out_channels,
        conv3_kernel_size=config.model.conv3_kernel_size,
        conv3_stride=config.model.conv3_stride,
        dropout_rate=config.model.dropout_rate,
        fc1_out_features=config.model.fc1_out_features,
    ).to(DEVICE)
    
    log.info(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")

    train_set, _ = corrupt_mnist()
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=config.training.batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = hydra.utils.instantiate(config.optimizer, params=model.parameters())

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(config.training.epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)

            if i % config.training.log_interval == 0:
                log.info(f"Epoch {epoch}, iter {i}, loss: {loss.item():.4f}, accuracy: {accuracy:.4f}")

    log.info("Training complete")

    # Save model if configured
    if config.training.save_model:
        # Use Hydra's output directory or the configured path
        import os
        model_path = config.training.model_path
        model_dir = os.path.dirname(model_path)
        if model_dir:  # Only create directory if path contains one
            os.makedirs(model_dir, exist_ok=True)
        torch.save(model.state_dict(), model_path)
        log.info(f"Model saved to {os.path.abspath(model_path)}")

    # Save plots if configured
    if config.training.save_plots:
        import os
        plot_path = config.training.plot_path
        plot_dir = os.path.dirname(plot_path)
        if plot_dir:  # Only create directory if path contains one
            os.makedirs(plot_dir, exist_ok=True)
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(statistics["train_loss"])
        axs[0].set_title("Train loss")
        axs[1].plot(statistics["train_accuracy"])
        axs[1].set_title("Train accuracy")
        fig.savefig(plot_path)
        log.info(f"Training statistics saved to {os.path.abspath(plot_path)}")


if __name__ == "__main__":
    train()

