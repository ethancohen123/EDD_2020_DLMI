import torch
from torch import nn, Tensor


class ConvBlock(nn.Module):
    """Basic convolutional block."""

    def __init__(self, in_channels, out_channels, norm='batch'):
        super().__init__()
        # choice of padding=1 keeps
        # feature map dimensions identical
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm == 'group':
            num_groups = out_channels // 8
            self.bn = nn.GroupNorm(num_groups, out_channels)
        elif norm is None:
            self.bn = nn.Identity()
        else:
            raise TypeError("Wrong type of normalization layer provided for ConvBlock")
        self.activation = nn.ReLU()

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class _DownBlock(nn.Module):
    """Contracting path segment.
    
    Downsamples using MaxPooling then applies ConvBlock.
    """

    def __init__(self, in_channels, out_channels, n_convs=2):
        super().__init__()
        layers = [
            ConvBlock(in_channels, out_channels)
        ] + [
            ConvBlock(out_channels, out_channels)
            for _ in range(n_convs-1)
        ]
        # maxpooling over patches of size 2
        self.mp = nn.MaxPool2d(2)
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mp(x)
        x = self.conv(x)
        return x


class _UpBlock(nn.Module):
    """Expansive path segment.
    
    Applies `~ConvBlock`, then upsampling deconvolution.
    """

    def __init__(self, in_channels, out_channels, n_convs=2, n_connect=2):
        """
        
        Parameters
        ----------
        n_connect : int
            Multiplicator for the number of input for the 1st convblock after
            the upsampling convolution (useful for skip connections).
        """
        super().__init__()
        layers = [
            # expects multiple of channels
            ConvBlock(n_connect * in_channels, in_channels)
        ] + [
            ConvBlock(in_channels, in_channels)
            for _ in range(n_convs-1)
        ]
        self.conv = nn.Sequential(*layers)
        # counts as one convolution
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         2, stride=2)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        z = torch.cat((skip, x), dim=1)
        z = self.conv(z)
        out = self.upconv(z)  # deconvolve
        return out


class UNet(nn.Module):
    """The U-Net architecture.
    
    See https://arxiv.org/pdf/1505.04597.pdf 
    """

    def __init__(self, num_channels: int=3, num_classes: int=2):
        """Initialize a U-Net.
        
        Parameters
        ----------
        num_channels : int
            Number of input channels.
        num_classes : int
            Number of output classes.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.in_conv = nn.Sequential(
            ConvBlock(num_channels, 64),
            ConvBlock(64, 64)
        )

        self.down1 = _DownBlock(64, 128)
        self.down2 = _DownBlock(128, 256)
        self.down3 = _DownBlock(256, 512)

        self.center = nn.Sequential(
            _DownBlock(512, 1024),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)  # upscale
        )

        # reminder: convolves then upsamples
        self.up1 = _UpBlock(512, 256)
        self.up2 = _UpBlock(256, 128)
        self.up3 = _UpBlock(128, 64)

        self.out_conv = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x: Tensor):
        x1 = self.in_conv(x)  # 64 * 1. * 1. ie 224
        x2 = self.down1(x1)  # 128 * 1/2 * 1/2
        x3 = self.down2(x2)  # 256 * 1/4 * 1/4
        x4 = self.down3(x3)  # 512 * 1/8 * 1/8
        x = self.center(x4)  # 512 * 1/8 * 1/8 ie 28
        x = self.up1(x, x4)  # 256 * 1/4 * 1/4 56
        x = self.up2(x, x3)  # 128 * 1/2 * 1/2 112
        x = self.up3(x, x2)
        z = torch.cat((x1, x), dim=1)
        out = self.out_conv(z)
        return out
