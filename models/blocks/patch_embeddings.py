import torch.nn as nn


class PatchEmbed3D(nn.Module):
    """ Volume to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,2,2).
        in_chans (int): Number of input volume channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 48.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, vol_size=(96, 96, 96), patch_size=(2, 2, 2), in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()
        patches_resolution = [vol_size[0] // patch_size[0], vol_size[1] // patch_size[1], vol_size[2] // patch_size[2]]
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, D, Wh, Ww)

        return x

class PatchEmbed3D(nn.Module):
    """ Volume to Patch Embedding.

    Args:
        patch_size (int): Patch token size. Default: (2,2,2).
        in_chans (int): Number of input volume channels. Default: 1.
        embed_dim (int): Number of linear projection output channels. Default: 48.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, vol_size=(96, 96, 96), patch_size=(2, 2, 2), in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()
        patches_resolution = [vol_size[0] // patch_size[0], vol_size[1] // patch_size[1], vol_size[2] // patch_size[2]]
        self.vol_size = vol_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1] * patches_resolution[2]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        # padding
        _, _, D, H, W = x.size()
        if W % self.patch_size[2] != 0:
            x = F.pad(x, (0, self.patch_size[2] - W % self.patch_size[2]))
        if H % self.patch_size[1] != 0:
            x = F.pad(x, (0, 0, 0, self.patch_size[1] - H % self.patch_size[1]))
        if D % self.patch_size[0] != 0:
            x = F.pad(x, (0, 0, 0, 0, 0, self.patch_size[0] - D % self.patch_size[0]))

        x = self.proj(x)  # B C D Wh Ww
        if self.norm is not None:
            D, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, D, Wh, Ww)

        return x


class PatchEmbedLarge(nn.Module):

    def __init__(self, in_chans=1, embed_dim=48, norm_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.pool = nn.MaxPool3d(2)
        self.conv1 = convLarge(in_chans, embed_dim // 16, nn.GELU, nn.LayerNorm, False)
        self.conv2 = convLarge(embed_dim // 16, embed_dim // 8, nn.GELU, nn.LayerNorm, False)
        self.conv3 = convLarge(embed_dim // 8, embed_dim // 4, nn.GELU, nn.LayerNorm, False)
        self.conv4 = convLarge(embed_dim // 4, embed_dim // 2, nn.GELU, nn.LayerNorm, False)
        self.conv5 = convLarge(embed_dim // 2, embed_dim, nn.GELU, nn.LayerNorm, True)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward function."""
        x = self.pool(x)
        x = self.conv1(x)  # B C Ws Wh Ww
        x = self.conv2(x)  # B C Ws Wh Ww
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        if self.norm is not None:
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.embed_dim, Ws, Wh, Ww)

        return x


class convLarge(nn.Module):
    def __init__(self, in_dim, out_dim, activate, norm, last=False):
        super().__init__()
        self.out_dim = out_dim
        self.conv1 = nn.Conv3d(in_dim, out_dim, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv3d(out_dim, out_dim, kernel_size=5, stride=1, padding=0)
        self.activate = activate()
        self.norm1 = norm(out_dim)
        self.last = last
        if not last:
            self.norm2 = norm(out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activate(x)
        # norm1
        Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.norm1(x)
        x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)

        x = self.conv2(x)
        if not self.last:
            x = self.activate(x)
            # norm2
            Ws, Wh, Ww = x.size(2), x.size(3), x.size(4)
            x = x.flatten(2).transpose(1, 2).contiguous()
            x = self.norm2(x)
            x = x.transpose(1, 2).contiguous().view(-1, self.out_dim, Ws, Wh, Ww)
        return x