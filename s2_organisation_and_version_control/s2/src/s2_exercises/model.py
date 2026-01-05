from torch import nn
import torch


class Model(nn.Module):
    """My awesome model."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""

        x = torch.relu(self.conv1(x))  # 1,32,N-2,D-2, where N = D = 28 for the dummy input it is height and width
        x = torch.max_pool2d(x, 2, 2)  # 1, 32, (N-2)/2, (D-2)/2
        x = torch.relu(self.conv2(x))  # 1, 64, (N-2)/2 - 2, (D-2)/2 -2
        x = torch.max_pool2d(x, 2, 2)  # 1, 64, (N-2)/2 -2 -2, (D-2)/2 -2 -2
        x = torch.relu(self.conv3(x))  # 1, 128, 3, 3
        x = torch.max_pool2d(x, 2, 2)  # 1, 128, 1, 1
        x = torch.flatten(x, 1)  # 1, 128
        x = self.dropout(x)
        return self.fc1(x)


if __name__ == "__main__":
    model = Model()
    print(f"Model architecture: {model}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    dummy_input = torch.randn(1, 1, 28, 28)  # Batch size, channels, height, width
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")

if __name__ == "__main__":
    model = Model()
    x = torch.rand(1, 1, 28, 28)
    print(f"Output shape of model: {model(x).shape}")
