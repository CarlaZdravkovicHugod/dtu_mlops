from torch import nn
import torch


class Model(nn.Module):
    """My awesome model."""

    def __init__(
        self,
        conv1_out_channels: int = 32,
        conv1_kernel_size: int = 3,
        conv1_stride: int = 1,
        conv2_out_channels: int = 64,
        conv2_kernel_size: int = 3,
        conv2_stride: int = 1,
        conv3_out_channels: int = 128,
        conv3_kernel_size: int = 3,
        conv3_stride: int = 1,
        dropout_rate: float = 0.5,
        fc1_out_features: int = 10,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, conv1_out_channels, conv1_kernel_size, conv1_stride)
        self.conv2 = nn.Conv2d(conv1_out_channels, conv2_out_channels, conv2_kernel_size, conv2_stride)
        self.conv3 = nn.Conv2d(conv2_out_channels, conv3_out_channels, conv3_kernel_size, conv3_stride)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(conv3_out_channels, fc1_out_features)

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
