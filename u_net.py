import torch
import torch.nn as nn
from torch.nn import init
import torchvision

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class DownBlock(nn.Module):
    """
    Down block that encapsulates one down-sampling step which consists of ConvBlock -> ConvBlock -> Pool
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        pre_pool_x = self.conv_block_1(x)
        pre_pool_x = self.conv_block_2(pre_pool_x)
        x = self.pool(pre_pool_x)
        return x, pre_pool_x


class UpBlock(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, upsampling_method="conv_transpose"):
        super().__init__()
        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """

        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UNet(nn.Module, initializer.IInitializable):
    """
    This is the UNet implementation referenced from:
    https://arxiv.org/abs/1505.04597

    Differences with the original implementation
    (1) In the original paper, cropping was done in the downsampling layers to include only the "real" pixels not the
        mirrored ones
    (2) Batch Norm was not widely known then, so as a result it is not used, though I have included it in my implementation
    (3) Upsampling can be done in two ways, through a transposed convolution, or bilinear upsampling
    """

    def __init__(self, n_input_channels=3, n_init_channels=64, depth=5, n_classes=2,
                 upsampling_method="conv_transpose"):

        """
        Arguments
        :param n_input_channels: number of input channels for the image
        :param n_init_channels: number of channels used for the output of the first ConvBlock, all proceeding channels
        used will be based on this number
        :param depth: number of layers to be used in the UNet
        :param upsampling_method: conv_transpose or bilinear
        """
        super().__init__()
        self.depth = depth
        if upsampling_method not in ['conv_transpose', 'bilinear']:
            raise ValueError(
                f"Please choose a valid upsampling method 'conv_transpose' or 'bilinear'"
            )

        down_blocks = []
        up_blocks = []

        # Construct downsampling layers
        in_channels = n_input_channels
        out_channels = n_init_channels
        for _ in range(depth - 1):
            down_block = DownBlock(in_channels, out_channels)
            down_blocks.append(down_block)
            in_channels = out_channels
            out_channels *= 2

        # Construct middle layers
        self.bridge = Bridge(in_channels, out_channels)

        # Construct upsampling layers
        in_channels = out_channels
        out_channels = int(out_channels / 2)

        for _ in range(depth - 1):
            up_block = UpBlock(in_channels, out_channels)
            up_blocks.append(up_block)
            in_channels = out_channels
            out_channels = int(out_channels / 2)

        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(up_blocks)
        # Construct final layer
        self.out = nn.Conv2d(in_channels, n_classes, kernel_size=1, stride=1)

        self.init_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)

    def init_params(self):
        for i, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, x):
        pre_pools = dict()
        for i, mod in enumerate(self.down_blocks, 1):
            x, pre_pool_x = mod(x)
            pre_pools[f"layer_{i}"] = pre_pool_x

        x = self.bridge(x)

        for i, mod in enumerate(self.up_blocks):
            key = f"layer_{self.depth - 1 - i}"
            x = mod(x, pre_pools[key])

        x = self.out(x)
        # del pre_pools
        return x

model = UNet().cuda()
inp = torch.rand((2, 3, 512, 512)).cuda()
out = model(inp)