from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


def dilated_convolution(module: nn.Conv2d, feature, dilation: int = 1):
    pd_size = _pair(dilation * module.padding[0])
    return F.conv2d(feature, module.weight, module.bias, module.stride,
                    pd_size, _pair(dilation), module.groups)


if __name__ == '__main__':
    import torch
    a = torch.randn(1, 1, 10, 10)
    conv = nn.Conv2d(1, 1, kernel_size=5, padding=2, dilation=2)
    print(conv(a).shape)
    print(dilated_convolution(conv, a, dilation=2).shape)
