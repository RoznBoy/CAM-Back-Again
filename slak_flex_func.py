# slak_flex_func.py
# coding: utf-8
"""
SLaK-Flex: SLaK의 십자형 ERF를 보정해서 RepLKNet의 ERF와 유사하게 만들기 위한 백본.

핵심 아이디어
-------------
- 원래 SLaK는 depthwise large kernel을 (k x 1) + (1 x k) strip conv로 구현 → ERF가 십자형.
- 여기서는 strip conv 두 개 + k x k depthwise conv를 동시에 두고
    out = alpha * (conv_h + conv_w) + beta * conv_square
  형태로 섞을 수 있게 함.
- alpha, beta 값을 조절하면서
    - alpha=1, beta=0  → SLaK-style 십자형 ERF
    - alpha=0, beta=1  → RepLKNet-style에 가까운 원형 ERF
    - 그 사이 값       → 중간 형태
- CAM-Back-Again의 ConvNeXt / RepLKNet과 동일한 구조:
    - 입력: 384x384
    - 마지막 feature map: 1024 채널, 12x12
    - Net2Head(class_n=200, unit_n=1024, size=12) 로 WSOL에 사용
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath

# -------------------------------------------------------------------------
# 0. ConvNeXt-style LayerNorm (channels-last 용, feature map에서 사용)
# -------------------------------------------------------------------------

class LayerNorm2d(nn.Module):
    """Channels-first (B, C, H, W)에 사용하는 LayerNorm2d."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        mean = x.mean(1, keepdim=True)
        var = (x - mean).pow(2).mean(1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

# -------------------------------------------------------------------------
# 1. SLaK-Flex large kernel depthwise conv
#    - strip conv (k x 1, 1 x k)
#    - square conv (k x k)
#    - 두 출력을 alpha, beta 비율로 섞음
# -------------------------------------------------------------------------

class SLaKFlexLargeKernelDW(nn.Module):
    """
    Strip conv + square conv 를 혼합하는 depthwise large kernel conv.

    Args:
        dim          : 채널 수 (depthwise conv라서 groups=dim)
        kernel_size  : 큰 커널 크기 (예: 51, 49, 47, 13)
        alpha_strip  : strip conv (k x 1 + 1 x k) 가중치
        beta_square  : square conv (k x k) 가중치
                       (alpha_strip + beta_square = 1 으로 설정하는 것을 권장하지만
                        코드 상으로는 자유롭게 둘 다 조절 가능)
    """
    def __init__(self, dim: int,
                 kernel_size: int,
                 alpha_strip: float = 1.0,
                 beta_square: float = 0.0):
        super().__init__()
        self.alpha_strip = alpha_strip
        self.beta_square = beta_square

        pad = kernel_size // 2

        # channel-wise strip convs
        self.conv_h = nn.Conv2d(
            dim, dim,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            groups=dim,
            bias=True,
        )
        self.conv_w = nn.Conv2d(
            dim, dim,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            groups=dim,
            bias=True,
        )

        # isotropic square depthwise conv
        self.conv_square = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=pad,
            groups=dim,
            bias=True,
        )

    @torch.no_grad()
    def set_mixing(self, alpha_strip: float, beta_square: float):
        """런타임에 ERF 형태를 바꾸고 싶을 때 호출."""
        self.alpha_strip = alpha_strip
        self.beta_square = beta_square

    def forward(self, x):
        """
        십자형 ERF와 원형 ERF를 가중합으로 섞음.

        out = alpha_strip * (conv_h(x) + conv_w(x)) + beta_square * conv_square(x)
        """
        out_strip = self.conv_h(x) + self.conv_w(x)
        out_square = self.conv_square(x)
        return self.alpha_strip * out_strip + self.beta_square * out_square

# -------------------------------------------------------------------------
# 2. ConvNeXt-style block + SLaK-Flex depthwise conv
# -------------------------------------------------------------------------

class SLaKFlexBlock(nn.Module):
    """
    ConvNeXt block 변형:
      - depthwise conv 부분을 SLaKFlexLargeKernelDW로 교체
      - 채널 확장/축소는 1x1 Linear 두 번으로 구현 (ConvNeXt와 동일)
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 51,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 1e-6,
        alpha_strip: float = 1.0,
        beta_square: float = 0.0,
    ):
        super().__init__()

        self.dwconv = SLaKFlexLargeKernelDW(
            dim=dim,
            kernel_size=kernel_size,
            alpha_strip=alpha_strip,
            beta_square=beta_square,
        )

        # ConvNeXt-style 채널 방향 LayerNorm + MLP
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(hidden_dim, dim)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                layer_scale_init_value * torch.ones(dim),
                requires_grad=True,
            )
        else:
            self.gamma = None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """
        shortcut = x

        # 1) depthwise large kernel conv (ERF를 결정하는 부분)
        x = self.dwconv(x)   # (B, C, H, W)

        # 2) 채널 축 방향 LN + MLP (ConvNeXt와 동일)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)

        # 3) residual + stochastic depth
        x = shortcut + self.drop_path(x)
        return x

# -------------------------------------------------------------------------
# 3. SLaK-Flex backbone (ConvNeXt-style 4-stage)
# -------------------------------------------------------------------------

class SLaKFlexBackbone(nn.Module):
    """
    ConvNeXt와 비슷한 4-stage 구조의 SLaK-Flex backbone.

    입력:  (B, 3, 384, 384)
    출력:  (B, dims[-1], 12, 12)   (384 / 32 = 12)
    """

    def __init__(
        self,
        in_chans: int = 3,
        depths=(3, 3, 27, 3),
        dims=(128, 256, 512, 1024),
        kernel_sizes=(51, 49, 47, 13),
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.3,
        alpha_strip: float = 1.0,
        beta_square: float = 0.0,
    ):
        super().__init__()
        self.depths = depths
        self.dims = dims

        # 3.1 stem & downsample layers
        self.downsample_layers = nn.ModuleList()
        # stem: 4x4 conv stride 4 → 384 → 96
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0]),
        )
        self.downsample_layers.append(stem)

        # stage 사이 downsample: 2x2 conv stride 2
        for i in range(3):
            down = nn.Sequential(
                LayerNorm2d(dims[i]),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(down)

        # 3.2 stochastic depth 비율 분배
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # 3.3 stages (각 stage는 여러 개의 SLaKFlexBlock 으로 구성)
        self.stages = nn.ModuleList()
        cur = 0
        for stage_idx in range(4):
            blocks = []
            for i in range(depths[stage_idx]):
                blocks.append(
                    SLaKFlexBlock(
                        dim=dims[stage_idx],
                        kernel_size=kernel_sizes[stage_idx],
                        mlp_ratio=mlp_ratio,
                        drop_path=dpr[cur + i],
                        alpha_strip=alpha_strip,
                        beta_square=beta_square,
                    )
                )
            cur += depths[stage_idx]
            self.stages.append(nn.Sequential(*blocks))

        # ImageNet-style 분류용 head (CAM-Back-Again에서는 대부분 사용하지 않고
        # Net2Head를 따로 붙여 WSOL에 활용)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(dims[-1], 1000)

    @torch.no_grad()
    def set_erf_mixing(self, alpha_strip: float, beta_square: float):
        """
        백본 전체에 대해 ERF mixing 비율을 바꾸고 싶을 때 사용.
        (실험 중간에 ERF 형상만 조절하고 싶으면 이 메서드를 호출하면 됨)
        """
        for stage in self.stages:
            for blk in stage:
                if isinstance(blk, SLaKFlexBlock):
                    blk.dwconv.set_mixing(alpha_strip, beta_square)

    def forward_features(self, x):
        # 입력: (B, 3, 384, 384)
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        # 출력: (B, C_last, 12, 12)
        return x

    def forward(self, x):
        # 일반 분류용 (WSOL에서는 Net2Head를 사용하는 것이 기본)
        x = self.forward_features(x)
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

# -------------------------------------------------------------------------
# 4. CAM-Back-Again에서 사용하는 Net2Head (ConvNeXt와 동일)
# -------------------------------------------------------------------------

class Net2Head(nn.Module):
    """
    CAM-Back-Again convnext_func.py 에서 사용하던 헤드와 동일.
    마지막 feature map (B, C, H, W)에 대해:
        - AvgPool(size x size)
        - FC -> class_n
        - Softmax
    """
    def __init__(self, class_n: int, unit_n: int, size: int):
        super().__init__()
        self.unit_n = unit_n
        self.size = size
        self.fc = nn.Linear(unit_n, class_n)
        self.cam = nn.AvgPool2d(size, size)

    def forward(self, x):
        x = self.cam(x)              # (B, C, 1, 1)
        x = x.view(x.size(0), -1)    # (B, C)
        x = self.fc(x)               # (B, class_n)
        return F.softmax(x, dim=1)

# -------------------------------------------------------------------------
# 5. 편의 생성 함수 & CAM-Back-Again용 build_model
# -------------------------------------------------------------------------

def create_SLaKFlex_B_384(
    drop_path_rate: float = 0.3,
    num_classes: int = 1000,
    alpha_strip: float = 1.0,
    beta_square: float = 0.0,
):
    """
    384 x 384 입력에 맞춘 base-size SLaK-Flex backbone.

    - depths: ConvNeXt-Base 스타일 [3, 3, 27, 3]
    - dims  : RepLKNet-31B와 비슷하게 [128, 256, 512, 1024]
    - kernel_sizes: RepLKNet와 비슷한 large kernel 크기 세트 (51, 49, 47, 13)
    """
    depths = (3, 3, 27, 3)
    dims = (128, 256, 512, 1024)
    kernel_sizes = (51, 49, 47, 13)
    model = SLaKFlexBackbone(
        in_chans=3,
        depths=depths,
        dims=dims,
        kernel_sizes=kernel_sizes,
        drop_path_rate=drop_path_rate,
        alpha_strip=alpha_strip,
        beta_square=beta_square,
    )
    # ImageNet head class 수 설정 (실제로는 Net2Head를 쓰기 때문에 큰 의미는 없음)
    model.head = nn.Linear(dims[-1], num_classes)
    return model


def build_model(model_config, fine_tuned_weights=None):
    """
    CAM-Back-Again 스타일 wrapper.

    model_config 예시:
        model_config = {
            "model_name": "SLaKFlex-B-384",
            "class_n": 200,
            "unit_n": 1024,
            "input_size": 384,
            "size": 12,
            "alpha_strip": 0.5,
            "beta_square": 0.5,
            ...
        }
    """
    model_name = model_config.get("model_name", "SLaKFlex-B-384")
    class_n = model_config["class_n"]
    unit_n = model_config["unit_n"]
    size = model_config["size"]

    alpha_strip = float(model_config.get("alpha_strip", 1.0))
    beta_square = float(model_config.get("beta_square", 0.0))

    if model_name == "SLaKFlex-B-384":
        backbone = create_SLaKFlex_B_384(
            drop_path_rate=0.3,
            num_classes=class_n,
            alpha_strip=alpha_strip,
            beta_square=beta_square,
        )
        # backbone의 ImageNet head는 사용하지 않고,
        # 마지막 feature map을 그대로 Net2Head 에 연결해서 WSOL에 사용
        backbone.head = nn.Identity()

        model = nn.Sequential(
            backbone,                      # (B, 3, 384, 384) -> (B, 1024, 12, 12)
            Net2Head(class_n, unit_n, size),
        )
    else:
        raise ValueError(f"Unknown SLaKFlex model_name: {model_name}")

    # fine-tuned weight 로드 (있으면)
    if fine_tuned_weights is not None:
        state = torch.load(fine_tuned_weights, map_location="cpu")
        if "model" in state:
            state = state["model"]
        if "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)
        print("loaded fine_tuned_weights for SLaK-Flex")

    return model
