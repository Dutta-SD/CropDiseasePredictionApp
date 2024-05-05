import torch
from torch import nn


class ConvADN(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel=2,
        stride=2,
        dilation=1,
        padding=0,
        p_drop=0.2,
        is_transpose: bool = False,
    ):
        super().__init__()
        self.model = nn.Sequential(
            (nn.Conv2d if not is_transpose else nn.ConvTranspose2d)(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel,
                stride=stride,
                dilation=dilation,
                padding=padding,
            ),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.InstanceNorm3d(num_features=out_channels),
        )

    def forward(self, x):
        return self.model(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            ConvADN(in_channels, 32, kernel=2, stride=2, padding=0),
            ConvADN(32, 64, kernel=2, stride=2, padding=0),
            ConvADN(64, 128, kernel=2, stride=2, padding=0),
            ConvADN(128, 256, kernel=2, stride=2, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            ConvADN(256, 128, kernel=2, stride=2, padding=0, is_transpose=True),
            ConvADN(128, 64, kernel=2, stride=2, padding=0, is_transpose=True),
            ConvADN(64, 32, kernel=2, stride=2, padding=0, is_transpose=True),
            ConvADN(32, out_channels, kernel=2, stride=2, padding=0, is_transpose=True),
        )
        self.output = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        return self.output(x)


class Autoencoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
