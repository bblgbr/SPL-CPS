import torch
import math
import warnings
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from compressai.models.google import JointAutoregressiveHierarchicalPriors
from net.patchmanager import PatchManager
from compressai.layers import GDN, MaskedConv2d
from compressai.ans import BufferedRansEncoder, RansDecoder

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution without padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride)

def conv2x2(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """2x2 convolution without padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=2, stride=stride)

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def Tconv2x2(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=stride)

def Tconv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 transposed convolution without padding."""
    return nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=stride)


class ResidualBlockWithStride(nn.Module):
    """Residual block with a stride on the first convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        stride (int): stride value (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 2, kernel_size: int = 3):
        super().__init__()
        if kernel_size == 2:
            self.conv1 = conv2x2(in_ch, out_ch, stride=stride)
        else:
            self.conv1 = conv1x1(in_ch, out_ch, stride=stride)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv1x1(out_ch, out_ch)
        self.gdn = GDN(out_ch)
        if stride != 1 or in_ch != out_ch:
            self.skip = conv2x2(in_ch, out_ch, stride=stride) if kernel_size == 2 else conv1x1(in_ch, out_ch, stride=stride)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.gdn(out)

        if self.skip is not None:
            identity = self.skip(x)

        out += identity
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with sub-pixel upsampling on the last convolution.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
        upsample (int): upsampling factor (default: 2)
    """

    def __init__(self, in_ch: int, out_ch: int, upsample: int = 2, kernel_size: int = 3):
        super().__init__()
        if kernel_size == 2:
            self.subpel_conv = Tconv2x2(in_ch, out_ch, upsample)
        else:
            self.subpel_conv = conv1x1(in_ch, out_ch, upsample)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv = conv1x1(out_ch, out_ch)
        self.igdn = GDN(out_ch, inverse=True)
        self.upsample = Tconv2x2(in_ch, out_ch, upsample) if kernel_size == 2 else conv1x1(in_ch, out_ch, upsample)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.subpel_conv(x)
        out = self.leaky_relu(out)
        out = self.conv(out)
        out = self.igdn(out)
        identity = self.upsample(x)
        out += identity
        return out


class ResidualBlock(nn.Module):
    """Simple residual block with two 1x1 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)
        self.conv2 = Tconv3x3(out_ch, out_ch)
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
    

def cal_receptive_field(conv_layer):
    l = 1
    s = 1
    for i in range(len(conv_layer)):
        l = l + (conv_layer[i][0] - 1) * s
        # print(l)
        s = s * conv_layer[i][1]
    return l

class CPS(JointAutoregressiveHierarchicalPriors):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(3, N, stride=2, kernel_size=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2, kernel_size=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2, kernel_size=2),
            ResidualBlock(N, N),
            conv2x2(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv2x2(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv2x2(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv1x1(N, N),
        )

        self.h_s = nn.Sequential(
            Tconv2x2(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            Tconv2x2(N, N * 3 // 2, stride=2),
            nn.LeakyReLU(inplace=True),
            conv1x1(N * 3 // 2, N * 2),
        )

        self.g_s = nn.Sequential(
            Tconv2x2(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2, kernel_size=2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2, kernel_size=2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, 3, 2, kernel_size=2),
        )
        self.device = 'cuda'
        self.N = N
        
        self.scale = 16 # downsample factor
        self.receptive_field = cal_receptive_field([[2,2], [3, 1], [3, 1], [2,2], [3, 1], [3, 1], [2,2], [3, 1], [3, 1], [2,2]]) # layer kernel size and stride
        
    
    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net
    
    def compress_patch(self, x, x_patch_size):
        self.patch_manger = PatchManager(x, x_patch_size, self.scale, self.receptive_field, self, device=self.device)
        # meta_y = self.patch_manger.encode_batch_processing(4)
        meta_y = self.patch_manger.encode()
        z = self.h_a(meta_y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(meta_y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(meta_y.size(0)):
            string = self._compress_ar(
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
        
    
    def decompress_patch(self, strings, shape, x_patch_size):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i : i + 1],
                params[i : i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))

        # meta_x = self.patch_manger.decode_batch_processing(y_hat, batch_size=4)
        meta_x = self.patch_manger.decode(y_hat)

        x_hat = meta_x.clamp_(0, 1)
        return {"x_hat": x_hat}
        
    
if __name__ == '__main__':
    model = CPS(192)
    img = torch.randn(1, 3, 256, 256)
    out = model(img)
    model.update(force=True)
    compress_byte = model.compress(img)
    decompress_byte = model.decompress(compress_byte['strings'], compress_byte['shape'])
    print(f"input size: {256}")
    print(f"forward output size: {out['x_hat'].shape}")
    print(f"compress output size: {decompress_byte['x_hat'].shape}")

