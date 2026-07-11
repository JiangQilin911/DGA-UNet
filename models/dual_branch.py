import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_models_pytorch import Unet

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x

class GlobalContext(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_context = nn.Conv2d(channels, channels, 1)
        self.conv_attn = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        context = self.conv_context(self.gap(x))
        attn = self.conv_attn(x)
        return x + context * attn

class DEMBranch(nn.Module):
    def __init__(self, in_channels=4, out_channels=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x

class DualBranchWrapper(nn.Module):
    def __init__(self, base_model, dem_in_channels=4, dem_out_channels=128,
                 reduction=16, dropout_rate=0.2):
        super().__init__()
        self.base_model = base_model
        self.dem_branch = DEMBranch(in_channels=dem_in_channels, out_channels=dem_out_channels)

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            if hasattr(self.base_model, 'encoder'):
                features = self.base_model.encoder(dummy)
                if isinstance(features, (list, tuple)):
                    rgb_channels = features[-1].shape[1]
                else:
                    rgb_channels = features.shape[1]
            else:
                raise AttributeError("Base model lacks 'encoder'")

        self.dem_proj = nn.Conv2d(dem_out_channels, rgb_channels, 1)

        self.gc_block = GlobalContext(rgb_channels, reduction)
        self.cbam = CBAM(rgb_channels, reduction)

        self.fusion_conv = nn.Conv2d(rgb_channels, rgb_channels, 1)
        self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        rgb = x[:, :3, :, :]
        dem = x[:, 3:, :, :]

        if hasattr(self.base_model, 'encoder'):
            rgb_features = self.base_model.encoder(rgb)
            if isinstance(rgb_features, (list, tuple)):
                rgb_feat = rgb_features[-1]
            else:
                rgb_feat = rgb_features
        else:
            raise NotImplementedError

        dem_feat = self.dem_branch(dem)
        dem_feat = self.dem_proj(dem_feat)
        if dem_feat.shape[2:] != rgb_feat.shape[2:]:
            dem_feat = F.interpolate(dem_feat, size=rgb_feat.shape[2:],
                                     mode='bilinear', align_corners=False)

        fused = rgb_feat + dem_feat

        fused = self.gc_block(fused)
        fused = self.cbam(fused)

        fused = self.fusion_conv(fused)

        # Dropout
        fused = self.dropout(fused)

        if isinstance(rgb_features, (list, tuple)):
            new_features = list(rgb_features)
            new_features[-1] = fused
            decoder_input = tuple(new_features)
        else:
            decoder_input = fused

        if hasattr(self.base_model, 'decoder'):
            decoder_out = self.base_model.decoder(decoder_input)
        else:
            decoder_out = fused

        if hasattr(self.base_model, 'segmentation_head'):
            out = self.base_model.segmentation_head(decoder_out)
        else:
            out = decoder_out
        return out


def create_model(num_classes=2, encoder_name='resnet50', encoder_weights='imagenet',
                 dropout_rate=0.2):
    base = Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                in_channels=3, classes=num_classes, activation=None)
    return DualBranchWrapper(base, dem_in_channels=4, dem_out_channels=128,
                             reduction=16, dropout_rate=dropout_rate)


if __name__ == "__main__":
    model = create_model(num_classes=2, dropout_rate=0.2)
    dummy = torch.randn(2, 7, 256, 256)
    with torch.no_grad():
        out = model(dummy)
    print(f"Group 3 output shape: {out.shape}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")