import math
from torch import nn
import torch

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.ca = ChannelAttention(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.bn1(self.conv1(x))
        out = self.relu(out)
        out = self.ca(out) * out
        out = self.bn2(self.conv2(out))
        out += residual
        return out


    
class RRDB(nn.Module):
    def __init__(self, channels, num_layers=3):
        super(RRDB, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(ResidualBlock(channels))
        self.conv = nn.Conv2d(num_layers * channels, channels, kernel_size=1)

    def forward(self, x):
        out = x
        features = []
        for layer in self.layers:
            out = layer(out)
            features.append(out)
        out = torch.cat(features, dim=1)
        out = self.conv(out)
        out += x
        return out

class RCANN(nn.Module):
    def __init__(self, channels):
        super(RCANN, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.ca = ChannelAttention(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.ca(out) * out
        out = self.conv2(out)
        out = out + x
        return out


class FSRCNN(nn.Module):
    def __init__(self, scale_factor, num_channels=1, d=56, s=12, m=4, rrdb_layers=3):
        super(FSRCNN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, d, kernel_size=5, padding=5//2),
            nn.PReLU(d)
        )
        self.mid_part = []
        for _ in range(m):
            self.mid_part.append(RRDB(d, num_layers=rrdb_layers))
        self.mid_part = nn.Sequential(*self.mid_part)
        self.last_part = nn.ConvTranspose2d(d, num_channels, kernel_size=9, stride=scale_factor, padding=9//2,
                                            output_padding=scale_factor-1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.first_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        for m in self.mid_part:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=math.sqrt(2/(m.out_channels*m.weight.data[0][0].numel())))
                nn.init.zeros_(m.bias.data)
        nn.init.normal_(self.last_part.weight.data, mean=0.0, std=0.001)
        nn.init.zeros_(self.last_part.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.mid_part(x)
        x = self.last_part(x)
        return x
