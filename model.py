import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


def init_weights_xavier(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def window_partition(x, window_size):
    #x: (B, C, H, W)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    # (数量，窗口大小，窗口大小，通道数)
    return windows


def window_reverse(windows, window_size, H, W):
    #windows: (num_windows*B, window_size, window_size, C)
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 -1
            layers.append(nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            # layers.append(nn.BatchNorm2d(out_channel))
            # layers.append(nn.InstanceNorm2d(out_channel, affine=True))
            pass
        if relu:
            # layers.append(nn.LeakyReLU(inplace=True, negative_slope=0.09))
            layers.append(nn.GELU())
            # layers.append(nn.Dropout(0.1))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, transpose=False, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            BasicConv(in_channels, mid_channels, kernel_size=3, stride=1, transpose=transpose),
            BasicConv(mid_channels, out_channels, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.double_conv(x)

class EBlock(nn.Module):
    def __init__(self, out_channel):
        super(EBlock, self).__init__()
        self.layers = DoubleConv(out_channel, out_channel)

    def forward(self, x):
        return self.layers(x)


class DBlock(nn.Module):
    def __init__(self, channel):
        super(DBlock, self).__init__()
        self.layers = DoubleConv(channel, channel)

    def forward(self, x):
        return self.layers(x)

# Channel Attention Moudle
class CAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        # Sigmoid 激活
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.avg_pool(x).view(b, c)
        max_out = self.max_pool(x).view(b, c)
        avg_out = self.fc(avg_out)
        max_out = self.fc(max_out)
        attention = avg_out + max_out
        attention = self.sigmoid(attention).view(b, c, 1, 1)
        return x * attention

# Spatial Attention Moudle
class SAM(nn.Module):
    def __init__(self, kernel_size=7):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # [B, 1, H, W]
        x_cat = torch.cat([avg_out, max_out], dim=1)  # [B, 2, H, W]
        attention = self.conv1(x_cat)  # [B, 1, H, W]
        attention = self.sigmoid(attention)
        return x * attention  # 应用到输入特征图: [B, C, H, W]

# Multi-scale Attention Module
class MAM(nn.Module):
    def __init__(self, in_channels, ker, drop_path=0.):
        super(MAM, self).__init__()

        pad = ker // 2
        self.conv11 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, groups=in_channels)
        self.conv13 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, ker), padding=(0, pad), groups=in_channels)
        self.conv31 = nn.Conv2d(in_channels, in_channels, kernel_size=(ker, 1), padding=(pad, 0), groups=in_channels)
        self.conv33 = nn.Conv2d(in_channels, in_channels, kernel_size=ker, padding=pad, groups=in_channels)

        self.sam11_13_31 = SAM()
        self.sam11_31_33 = SAM()
        self.sam13_31_33 = SAM()

        self.reduce = nn.Conv2d(3 * in_channels, in_channels, kernel_size=1, padding=0)
        self.cam = CAM(in_channels*3)
        self.dropout = nn.Dropout(drop_path)
        # self.final_conv = nn.Conv2d(in_channels, base_channel, kernel_size=1)

    def forward(self, x):
        shoutcut = x
        out11 = self.conv11(x)
        out13 = self.conv13(x)
        out31 = self.conv31(x)
        out33 = self.conv33(x)

        out111331 = torch.cat([out11, out13, out31], dim=1)
        out113133 = torch.cat([out11, out31, out33], dim=1)
        out133133 = torch.cat([out13, out31, out33], dim=1)

        out_sam111331 = self.sam11_13_31(out111331)
        out_sam113133 = self.sam11_31_33(out113133)
        out_sam133133 = self.sam13_31_33(out133133)

        out_sam = out_sam111331+out_sam113133+out_sam133133
        out_cam = self.cam(out_sam)
        out = self.reduce(out_cam)
        out = self.dropout(out)

        return out+shoutcut


class MAMgroup(nn.Module):
    def __init__(self, in_channel,ker, drop_path):
        super().__init__()
        depths=[4]
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.mam1 = MAM(in_channel,ker, dpr[0])
        self.mam2 = MAM(in_channel,ker,dpr[1])
        self.mam3 = MAM(in_channel,ker,dpr[2])
        self.mam4 = MAM(in_channel,ker,dpr[3])


    def forward(self, x):
        shortcut = x
        mam1 = self.mam1(x)
        mam2 = self.mam2(mam1)
        mam3 = self.mam3(mam2+shortcut)
        mam4 = self.mam4(mam1+mam3)
        out = mam2+mam4+shortcut

        return out

class SAPB(nn.Module):
    def __init__(self, window_size, dim, num_heads, qkv_bias, attn_drop=0.):
        super(SAPB, self).__init__()
        self.num_heads = num_heads
        self.window_size = window_size
        self.dim = dim
        self.kv = nn.Linear(dim, dim*2, qkv_bias)
        self.q = nn.Linear(dim, dim, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.project_out = nn.Linear(dim, dim, qkv_bias)

        self.spectral_pos_embed = nn.Parameter(torch.zeros(window_size * window_size, dim))
        nn.init.trunc_normal_(self.spectral_pos_embed, std=.02)

    def forward(self, x):
        b, n, c = x.shape
        shortcut = x

        x = x + self.spectral_pos_embed.unsqueeze(0)  # [1, 64, c] → [b, 64, c]

        kv = self.kv(x)
        k, v = kv.chunk(2, dim=2)
        q = self.q(x)

        q = q.reshape(-1, self.num_heads, q.shape[-1] // self.num_heads, n).permute(0, 1, 3, 2).contiguous()
        k = k.reshape(-1, self.num_heads, k.shape[-1] // self.num_heads, n).permute(0, 1, 3, 2).contiguous()
        v = v.reshape(-1, self.num_heads, v.shape[-1] // self.num_heads, n).permute(0, 1, 3, 2).contiguous()

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        v = F.normalize(v, dim=-1)

        attn = (k.transpose(-2, -1) @ q)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (v @ attn)

        out = out.reshape(b, n, c)
        out = self.project_out(out + shortcut)
        # out = self.project_out(out)
        return out



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# STB
class STB(nn.Module):
    def __init__(self,window_size, dim, num_heads, qkv_bias, drop=0., act_layer=nn.GELU, attn_drop=0., drop_path=0., mlp_ratio=4.):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.sapb = SAPB(window_size, dim, num_heads, qkv_bias, attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        B,C,H,W =  x.shape
        shoutcut = x
        x_temp = x.permute(0,2,3,1)
        norm1 = self.norm1(x_temp)

        x1_windows = window_partition(x=norm1, window_size=self.window_size)
        B1,H1,W1,C1 = x1_windows.shape
        x1_windows = x.reshape(-1, H1*W1, C1)
        sapb_result = self.sapb(x1_windows)
        sapb_result = window_reverse(sapb_result, self.window_size, H, W)
        sapb_result = sapb_result.permute(0,3,1,2)

        x2 = self.drop_path(sapb_result) + shoutcut
        x2_temp = x2.permute(0, 2, 3, 1)
        shoutcut_2 = x2
        norm2 = self.norm2(x2_temp)
        mlp_result = self.mlp(norm2)
        mlp_result = mlp_result.permute(0,3,1,2)
        result = shoutcut_2 + self.drop_path(mlp_result)

        return  result

# SRFB
class SRFB(nn.Module):
    def __init__(self, window_size, dim,num_head=8, mlp_ratio=4,drop_path=0.0, attn_drop=0., qkv_bias=False,):
        super().__init__()
        depths = [4]  # 模型总层数
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        self.stb1=STB(window_size=window_size,dim=dim,num_heads=num_head,qkv_bias=qkv_bias,mlp_ratio=mlp_ratio,drop_path=dpr[0],attn_drop=attn_drop)
        self.stb2=STB(window_size=window_size,dim=dim,num_heads=num_head,qkv_bias=qkv_bias,mlp_ratio=mlp_ratio,drop_path=dpr[1],attn_drop=attn_drop)
        self.stb3=STB(window_size=window_size,dim=dim,num_heads=num_head,qkv_bias=qkv_bias,mlp_ratio=mlp_ratio,drop_path=dpr[2],attn_drop=attn_drop)
        self.stb4=STB(window_size=window_size,dim=dim,num_heads=num_head,qkv_bias=qkv_bias,mlp_ratio=mlp_ratio,drop_path=dpr[3],attn_drop=attn_drop)
    def forward(self, x):
        shortcut =x
        stb1 = self.stb1(x)
        stb2 = self.stb2(stb1)
        stb3 = self.stb3(stb2+shortcut)
        stb4 = self.stb4(stb1+stb3)
        out = stb4+stb2+shortcut

        return out


class HSINet(nn.Module):
    def __init__(self, window_size=8,
               in_channel=31,
               base_channel=96,
               num_heads=8,
               mlp_ratio=4,
               drop_path=0.1,
               attn_drop=0.,
               qkv_bias=True):
        super().__init__()
        #1, 1/2图的三个卷积
        self.image_2_conv = nn.Conv2d(in_channel, base_channel*2, kernel_size=3, padding=1)
        self.image_1_conv = nn.Conv2d(in_channel, base_channel, kernel_size=3, padding=1)

        # ---------- Encoder ----------
        self.enc1_conv = BasicConv(in_channel, base_channel, 3, 1)
        self.enc1 = EBlock(base_channel)

        self.down1 = BasicConv(base_channel, base_channel*2, 3, 2)
        self.enc2 = EBlock(base_channel*2)

        self.down2 = BasicConv(base_channel*2, base_channel*4, 3, 2)
        self.enc3 = EBlock(base_channel*4)

        self.down3 = BasicConv(base_channel*4, base_channel*8, 3, 2)

        # ---------- Skip Large Kernel ----------
        self.pool = nn.AvgPool2d(2)
        self.reduce1 = nn.Conv2d(base_channel*3, base_channel*2, 3,1,1,bias=True)
        self.reduce2 = nn.Conv2d(base_channel * 6, base_channel * 4, 3, 1, 1, bias=True)
        self.skip_lk1 = MAMgroup(base_channel, 63, drop_path)
        self.skip_lk2 = MAMgroup(base_channel*2, 31,drop_path)
        self.skip_lk3 = MAMgroup(base_channel*4, 15, drop_path)

        # ---------- Bottleneck ----------
        self.bottleneck = EBlock(base_channel*8)
        self.spectral_block = SRFB(
            dim=base_channel*8,
            window_size=window_size,
            num_head=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path=drop_path,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop
        )

        # ---------- Decoder ----------
        self.up3 = BasicConv(base_channel*8, base_channel*4, 4, 2, transpose=True)
        self.dec3 = DBlock(base_channel*4)

        self.up2 = BasicConv(base_channel*4, base_channel*2, 4, 2, transpose=True)
        self.dec2 = DBlock(base_channel*2)

        self.up1 = BasicConv(base_channel*2, base_channel, 4, 2, transpose=True)
        self.dec1 = DBlock(base_channel)

        # ---------- Output ----------
        self.out_conv = nn.Conv2d(base_channel, in_channel, 3, padding=1)

    def forward(self, x):
        # -------- Encoder --------
        e1 = self.enc1(self.enc1_conv(x))    # 64×64
        e2 = self.enc2(self.down1(e1))       # 32×32
        e3 = self.enc3(self.down2(e2))       # 16×16


        # -------- Skip Large Kernel --------
        s1 = self.skip_lk1(e1)
        # s1 = e1
        s2 = self.skip_lk2(self.reduce1(torch.cat([self.pool(s1),e2],dim=1)))
        # s2 = e2
        s3 = self.skip_lk3(self.reduce2(torch.cat([self.pool(s2),e3],dim=1)))
        # s3 = e3

        # -------- Bottleneck --------
        b = self.down3(e3)                   # 8×8
        b = self.bottleneck(b)
        b = self.spectral_block(b)

        # -------- Decoder --------
        d3 = self.up3(b) + s3                # 16×16
        d3 = self.dec3(d3)

        d2 = self.up2(d3) + s2               # 32×32
        d2 = self.dec2(d2)

        d1 = self.up1(d2) + s1               # 64×64
        d1 = self.dec1(d1)

        return self.out_conv(d1)

if __name__ == '__main__':

    import torch
    from thop import profile
    from thop import clever_format

    model = HSINet()

    # 格式：[batch, channels, height, width]
    dummy_input = torch.randn(1, 31, 64, 64)

    flops, params = profile(model, inputs=(dummy_input,))

    flops_formatted, params_formatted = clever_format([flops, params], "%.3f")

    # 5. 打印结果
    print("=" * 50)
    print(f"模型总参数量 (Params): {params_formatted}")
    print(f"模型总计算量 (FLOPs): {flops_formatted}")
    print("=" * 50)