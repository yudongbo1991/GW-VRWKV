import torch.nn as nn
import torch.nn.init as init
import torch
import math
import torch.nn.functional as F
from einops import rearrange
from .model.vrwkv_IP import HSI_RWKV, VRWKV_SpatialMix, VRWKV_ChannelMix
from typing import Optional
# from .Refine_module import SSDP_Stem,SSC_Stem,SSC_Stem_v2,SC_GSP_Stem,SC_GSP_Stem_V2,WaveletLowGuideHighFreq,WaveletDynamicHighFreq,WaveletDynamicRefiner
# from .Refine_module import HCA,HFE,Lightweight_Spectral_Stem,Adaptive_Context_Refiner,RWKVFeatureFusion,LateralInhibitionFusion,RWKV_SSFM


def build_rwkv_stack(hidden_dim: int, group_num: int, spec: list) -> nn.Sequential:
    """
    Build nn.Sequential for rwkv trunk from a list of ops.

    Each op is a dict-like object, e.g.
      {"type": "rwkv", "num_blocks": [1]}
      {"type": "pool", "mode": "max", "k": 2, "s": 2}
      {"type": "norm", "name": "groupnorm"}
      {"type": "act", "name": "silu"}
    """
    if spec is None or len(spec) == 0:
        raise ValueError("rwkv_spec is empty. Please provide a non-empty spec list.")

    layers = []
    for op in spec:
        if not isinstance(op, dict):
            raise TypeError(f"Each rwkv op must be a dict, got: {type(op)}")

        t = str(op.get("type", "")).lower().strip()
        if t == "rwkv":
            num_blocks = op.get("num_blocks", [1])
            layers.append(HSI_RWKV(dim=hidden_dim, num_blocks=num_blocks))

        elif t == "pool":
            mode = str(op.get("mode", "max")).lower().strip()
            k = int(op.get("k", 2))
            s = int(op.get("s", k))
            if mode == "max":
                layers.append(nn.MaxPool2d(kernel_size=k, stride=s))
            elif mode == "avg":
                layers.append(nn.AvgPool2d(kernel_size=k, stride=s))
            else:
                raise ValueError(f"Unknown pool mode: {mode}")

        elif t == "norm":
            name = str(op.get("name", "groupnorm")).lower().strip()
            if name == "groupnorm":
                layers.append(nn.GroupNorm(group_num, hidden_dim))
            else:
                raise ValueError(f"Unknown norm: {name}")

        elif t == "act":
            name = str(op.get("name", "silu")).lower().strip()
            if name == "silu":
                layers.append(nn.SiLU())
            elif name == "relu":
                layers.append(nn.ReLU(inplace=True))
            elif name == "gelu":
                layers.append(nn.GELU())
            else:
                raise ValueError(f"Unknown activation: {name}")

        else:
            raise ValueError(f"Unknown op type: {t}. Supported: rwkv/pool/norm/act")

    return nn.Sequential(*layers)


class HiRWKV(nn.Module):
    """
    HiRWKV with configurable RWKV trunk.

    Parameters
    ----------
    rwkv_spec: list[dict] | None
        If provided, build self.rwkv from this spec.
        If None, fall back to default HANCHUAN-like stack (backward compatible).
    dataset_name: str | None
        Optional tag for logging / debugging.
    """
    def __init__(
        self,
        in_channels=128,
        hidden_dim=128,
        num_classes=10,
        group_num=1,
        rwkv_spec=None,
        dataset_name: Optional[str] = None
            # dataset_name: str | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )

        # Default (backward compatible): your original HANCHUAN trunk
        default_spec = [
            {"type": "rwkv", "num_blocks": [1]},
            {"type": "pool", "mode": "max", "k": 2, "s": 2},
            {"type": "act", "name": "silu"},
            {"type": "rwkv", "num_blocks": [1]},
            {"type": "pool", "mode": "max", "k": 2, "s": 2},
            {"type": "act", "name": "silu"},
            {"type": "rwkv", "num_blocks": [1]},
        ]
        spec = default_spec if rwkv_spec is None else rwkv_spec

        self.rwkv = build_rwkv_stack(hidden_dim=hidden_dim, group_num=group_num, spec=spec)

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.refine = WaveletAttentionRefiner(dim=hidden_dim)
    def forward(self, x):
        x = self.patch_embedding(x)
        x = self.rwkv(x)
        x = self.refine(x)
        logits = self.cls_head(x)
        return logits

################################################################

def window_partition(x, window_size: int):
    B, C, H, W = x.shape
    x = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=window_size, p2=window_size)
    return x


def window_reverse(windows, window_size: int, H: int, W: int):
    # windows shape: [(B*h*w), (p1*p2), C]
    x = rearrange(windows, '(b h w) (p1 p2) c -> b c (h p1) (w p2)',
                  h=H // window_size, w=W // window_size, p1=window_size, p2=window_size)
    return x

def window_partition_v2(x, window_size):
    """
    将 [B, C, H, W] 分割为 [B * num_windows, window_size * window_size, C]
    """
    B, C, H, W = x.shape
    # x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.reshape(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    windows = windows.view(-1, window_size * window_size, C)
    return windows

def window_reverse_v2(windows, window_size, H, W):
    """
    将窗口还原回全图
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    x = x.view(B, -1, H, W)
    return x

#########################new rwkv mode

class GWRWKV(nn.Module):
    """
    HiRWKV with configurable RWKV trunk.

    Parameters
    ----------
    rwkv_spec: list[dict] | None
        If provided, build self.rwkv from this spec.
        If None, fall back to default HANCHUAN-like stack (backward compatible).
    dataset_name: str | None
        Optional tag for logging / debugging.
    """

    def __init__(
            self,
            in_channels=128,
            hidden_dim=128,
            num_classes=10,
            group_num=1,
            rwkv_spec=None,
            dataset_name: Optional[str] = None,
            winsize = 16
            # dataset_name: str | None = None,
    ):
        super().__init__()
        self.dataset_name = dataset_name

        self.patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
        )
        self.w_size = winsize
        print("#####################")
        print(self.w_size)
        # default_spec1 = [
        #     {"type": "rwkv", "num_blocks": [1]},
        #     {"type": "pool", "mode": "max", "k": 2, "s": 2},
        #     {"type": "act", "name": "silu"},
        # ]
        #
        # self.rwkv1 = build_rwkv_stack(hidden_dim=hidden_dim, group_num=group_num, spec=default_spec1)
        #
        # default_spec2 = [
        #     {"type": "rwkv", "num_blocks": [1]},
        #     {"type": "pool", "mode": "max", "k": 2, "s": 2},
        #     {"type": "act", "name": "silu"},
        # ]
        # self.w_size = 32
        #
        # self.rwkv2 = build_w_rwkv_stack(hidden_dim=hidden_dim, group_num=group_num, spec=default_spec2,
        #                                 w_size=self.w_size)
        #
        # default_spec3 = [
        #     # {"type": "rwkv", "num_blocks": [1]},
        #     # {"type": "pool", "mode": "max", "k": 2, "s": 2},
        #     # {"type": "act", "name": "silu"},
        #     {"type": "rwkv", "num_blocks": [1]},
        # ]
        #
        # self.rwkv3 = build_rwkv_stack(hidden_dim=hidden_dim, group_num=group_num, spec=default_spec3)

        self.cls_head = nn.Sequential(
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(group_num, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=3, stride=1, padding=1),
        )
        # self.sefe = HSI_MS2_Stem(in_channels=hidden_dim, hidden_dim=hidden_dim)

        spec = default_spec if rwkv_spec is None else rwkv_spec
        self.rwkv = build_new_rwkv_stack(hidden_dim=hidden_dim, group_num=group_num, spec=spec, w_size=self.w_size)
        # self.rwkv = build_multi_scale_stack(hidden_dim=hidden_dim, group_num=group_num, spec=spec, w_size=self.w_size)

        # self.cls_head = nn.Sequential(
        #     nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=1, stride=1, padding=0),
        #     nn.GroupNorm(group_num, hidden_dim),
        #     nn.SiLU(),
        #     nn.Conv2d(in_channels=hidden_dim, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
        # )
        # self.refine = RefinedPWR(hidden_dim=hidden_dim)
        # self.refine = RefinedPWR_v4(hidden_dim=hidden_dim)

        # self.sefe = HSI_B1_Aware_Stem(in_channels=hidden_dim, hidden_dim=hidden_dim)
        # self.sefe = SSDP_Stem(in_channels=hidden_dim, hidden_dim=hidden_dim)
        # self.sefe = SSC_Stem(in_channels=hidden_dim, hidden_dim=hidden_dim)
        # self.sefe = SSC_Stem_422(in_channels=hidden_dim, hidden_dim=hidden_dim)
        # self.sefe = SSC_Stem_v2(in_channels=hidden_dim, hidden_dim=hidden_dim)
        # self.sefe = SC_GSP_Stem(in_channels=hidden_dim, hidden_dim=hidden_dim)
        # self.refine = EnhancedWaveletRefiner_421(dim=hidden_dim)
        # self.refine = WaveletDynamicRefiner(dim=hidden_dim)
        # self.refine2 = HFE(dim=hidden_dim)

    def forward(self, x):
        x = self.patch_embedding(x)
        B, C, H, W = x.shape
        # x, (H, W) = self.sefe(x)
        # x= self.sefe(x)
        x = self.rwkv(x)
        ###########33
        # B, C, H, W = x.shape
        #
        # pad_h = (self.w_size - H % self.w_size) % self.w_size
        # pad_w = (self.w_size - W % self.w_size) % self.w_size
        # x = F.pad(x, (0, pad_w, 0, pad_h))

        # x = self.rwkv2(x)

        # 去除 Padding
        # x = x[:, :, :H, :W]
        #######################3
        # x = self.rwkv3(x)
        # x = self.refine(x, H, W)
        # x = self.refine(x)
        # x = self.refine2(x)
        logits = self.cls_head(x)
        return logits


def build_new_rwkv_stack(hidden_dim: int, group_num: int, spec: list, w_size: int) -> nn.Sequential:
    if spec is None:
        raise ValueError("spec cannot be None")

    layers = []

    for op in spec:
        t = str(op.get("type", "")).lower().strip()

        if t == "swrwkv":
            # 获取当前阶段的 Block 数量，例如 [4]
            num_blocks_list = op.get("num_blocks", [1])
            num_blocks = num_blocks_list[0]

            block_list = []
            for i in range(num_blocks):
                block_list.append(
                    Swin_RWKV_Block_Double(
                        n_embd=hidden_dim,
                        n_layer=num_blocks,
                        layer_id=i,
                        window_size=w_size
                    )
                )

            layers.append(nn.Sequential(*block_list))

        elif t == "rwkv":
            num_blocks = op.get("num_blocks", [1])
            layers.append(HSI_RWKV(dim=hidden_dim, num_blocks=num_blocks))
        elif t == "hrwkv":
            num_blocks_list = op.get("num_blocks", [1])
            num_blocks = num_blocks_list[0]

            block_list = []
            for i in range(num_blocks):
                block_list.append(
                    Hybrid_RWKV_Block(
                        n_embd=hidden_dim,
                        n_layer=num_blocks,
                        layer_id=i,
                        window_size=w_size
                    )
                )
            layers.append(nn.Sequential(*block_list))

        elif t == "hrwkvno":
            num_blocks_list = op.get("num_blocks", [1])
            num_blocks = num_blocks_list[0]

            block_list = []
            for i in range(num_blocks):
                block_list.append(
                    Hybrid_RWKV_Block_no(
                        n_embd=hidden_dim,
                        n_layer=num_blocks,
                        layer_id=i,
                        window_size=w_size
                    )
                )
            layers.append(nn.Sequential(*block_list))

        elif t == "stem":
            layers.append(IPM(in_channels=hidden_dim, hidden_dim=hidden_dim))

        elif t == "pool":
            # 下采样层：池化后通常会接一个 Norm 和 Act
            k = op.get("k", 2)
            layers.append(nn.MaxPool2d(kernel_size=k, stride=op.get("s", k), padding=0, ceil_mode=True))

        elif t == "norm":
            layers.append(nn.GroupNorm(group_num, hidden_dim))

        elif t == "act":
            name = op.get("name", "silu")
            layers.append(nn.SiLU() if name == "silu" else nn.GELU())

    return nn.Sequential(*layers)

class Hybrid_RWKV_Block(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, window_size=11):
        super().__init__()
        self.window_size = window_size
        # 确保 n_layer 是 list，防止 HSI_RWKV 报错
        hsi_num_blocks = n_layer if isinstance(n_layer, list) else [n_layer]

        # --- 局部分支：该模块内部自带了常规窗口和偏移窗口（Shift）两次处理
        self.local_branch = Swin_RWKV_Block_Double(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            window_size=window_size
        )

        # --- 全局分支：直接调用 HSI_RWKV ---
        # 假设 HSI_RWKV 内部执行的是全图序列建模
        self.global_branch = HSI_RWKV(
            dim=n_embd,
            num_blocks=hsi_num_blocks
        )

        # 可学习融合参数：
        # self.fusion_weights = nn.Parameter(torch.tensor([1.0, 3.0]))

        self.fusion = BGFM(dim=n_embd)
    def forward(self, x):
        # 保持原始输入用于残差
        identity = x

        # 1. 计算归一化权重
        # weights = torch.softmax(self.fusion_weights, dim=0)

        # 2. 并行分支计算
        # local_branch 输入输出均为 [B, C, H, W]
        out_local = self.local_branch(x)

        # global_branch 输入输出均为 [B, C, H, W]
        # 注意：如果 HSI_RWKV 内部没有处理内存连续性，这里建议加 .contiguous()
        out_global = self.global_branch(x)

        # 3. 加权融合
        # a * Local_Double + b * Global
        # out = out_local * weights[0] + out_global * weights[1]
        out = self.fusion(out_local, out_global)
        # out = self.hca1(out_local) * weights[0] + self.hca2(out_global) * weights[1]
        # out = self.local_branch(self.global_branch(x))
        return out+identity
class Hybrid_RWKV_Block_no(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, window_size=11):
        super().__init__()
        self.window_size = window_size
        # 确保 n_layer 是 list，防止 HSI_RWKV 报错
        hsi_num_blocks = n_layer if isinstance(n_layer, list) else [n_layer]

        # --- 局部分支：直接调用你的 Swin_RWKV_Block_Double ---
        # 该模块内部自带了常规窗口和偏移窗口（Shift）两次处理
        self.local_branch = Swin_RWKV_Block_Double(
            n_embd=n_embd,
            n_layer=n_layer,
            layer_id=layer_id,
            window_size=window_size
        )

        # --- 全局分支：直接调用HSI_RWKV ---
        # 假设 HSI_RWKV 内部执行的是全图序列建模
        self.global_branch = HSI_RWKV(
            dim=n_embd,
            num_blocks=hsi_num_blocks  # 或者根据定义传入具体的层数
        )

        # 可学习融合参数：
        self.fusion_weights = nn.Parameter(torch.tensor([1.0, 3.0]))


    def forward(self, x):
        # 保持原始输入用于残差
        identity = x

        # 1. 计算归一化权重
        weights = torch.softmax(self.fusion_weights, dim=0)

        # 2. 并行分支计算
        # local_branch 输入输出均为 [B, C, H, W]
        out_local = self.local_branch(x)

        # global_branch 输入输出均为 [B, C, H, W]

        out_global = self.global_branch(x)

        # 3. 加权融合
        # a * Local_Double + b * Global
        out = out_local * weights[0] + out_global * weights[1]
        # out = self.fusion(out_local, out_global)
        # out = self.hca1(out_local) * weights[0] + self.hca2(out_global) * weights[1]

        return out+identity
class Swin_RWKV_Block_Double(nn.Module):
    def __init__(self, n_embd, n_layer, layer_id, window_size=32):
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2

        # --- 第一组算子：标准窗口 ---
        self.ln1 = nn.LayerNorm(n_embd)
        self.att1 = VRWKV_SpatialMix(n_embd=n_embd, n_layer=n_layer, layer_id=layer_id,edge_gate=False)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffn1 = VRWKV_ChannelMix(n_embd=n_embd, n_layer=n_layer, layer_id=layer_id,whiten=False)

        # # # --- 第二组算子：偏移窗口 ---
        self.ln3 = nn.LayerNorm(n_embd)
        self.att2 = VRWKV_SpatialMix(n_embd=n_embd, n_layer=n_layer, layer_id=layer_id,edge_gate=False)
        self.ln4 = nn.LayerNorm(n_embd)
        self.ffn2 = VRWKV_ChannelMix(n_embd=n_embd, n_layer=n_layer, layer_id=layer_id,whiten=False)

    def _process_window(self, x, att_op, ffn_op, ln1_op, ln2_op, shift=False):
        B, C, H, W = x.shape
        ws = self.window_size

        # 1. Padding 逻辑 (保持偶数适配)
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        pad_top, pad_left = pad_h // 2, pad_w // 2
        pad_bottom, pad_right = pad_h - pad_top, pad_w - pad_left
        x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode='reflect')
        _, _, H_pad, W_pad = x.shape

        # 2. Shift 逻辑
        if shift:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # 3. 窗口切分与 RWKV 计算
        x_win = window_partition_v2(x, ws)
        # x_win = window_partition(x, ws).contiguous()  # 加上 .contiguous()
        patch_res = (ws, ws)
        x_win = x_win + att_op(ln1_op(x_win), patch_resolution=patch_res)
        x_win = x_win + ffn_op(ln2_op(x_win), patch_resolution=patch_res)

        # 4. 还原
        out = window_reverse_v2(x_win, ws, H_pad, W_pad)

        # 5. 反向 Shift
        if shift:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # 6. 裁剪
        return out[:, :, pad_top:pad_top + H, pad_left:pad_left + W].contiguous()

    def forward(self, x):
        # 第一次：常规窗口处理
        x = self._process_window(x, self.att1, self.ffn1, self.ln1, self.ln2, shift=False)
        # 第二次：偏移窗口处理 (Shift)
        x = self._process_window(x, self.att2, self.ffn2, self.ln3, self.ln4, shift=True)
        # x = self._process_window(x, self.att1, self.ffn1, self.ln1, self.ln2, shift=True)
        return x



class BGFM(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # 局部引导：空间锐化
        self.spatial_refiner = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        # 全局引导：光谱校准
        self.spectral_refiner = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        # 可学习的融合因子，初始化为0，保证刚开始训练时等同于直接相加，不破坏原有性能
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, f_local, f_global):
        # 1. 局部指导全局 (锐化边界)
        # 提取局部特有的空间高频信息
        spatial_res = self.spatial_refiner(f_local - f_global)
        f_g_new = f_global + self.alpha * spatial_res

        # 2. 全局指导局部 (校准光谱分布)
        # 利用全局平滑的语义信息作为权重
        spectral_weight = self.spectral_refiner(f_global)
        f_l_new = f_local * (1 + self.beta * spectral_weight)

        # 3. 简单的求和融合
        return f_g_new + f_l_new



class IPM(nn.Module):
    """
    Spectral-Spatial Continuity Stem (SSC-Stem)
    融合了坐标感知、全局光谱调制与空间连续性约束
    """

    def __init__(self, in_channels, hidden_dim, groups=8):
        super().__init__()

        # 1. 坐标生成 (全局空间先验)
        self.pos_embed = nn.Sequential(
            nn.Conv2d(2, hidden_dim // 4, 1),
            nn.GroupNorm(1, hidden_dim // 4),
            nn.GELU()
        )

        # 2. 基础多尺度空间路径
        self.proj = nn.Conv2d(in_channels, hidden_dim, 3, padding=1)
        self.spatial_dw = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=2, dilation=2, groups=hidden_dim),
            nn.GroupNorm(groups, hidden_dim),
            nn.GELU()
        )

        # 3. 光谱-坐标交互模块 (创新：根据位置校准全局感知)
        self.global_spectral = nn.AdaptiveAvgPool2d(1)
        self.coord_spectral_gate = nn.Sequential(
            nn.Conv2d(hidden_dim + hidden_dim // 4, hidden_dim, 1),
            nn.GroupNorm(groups, hidden_dim),
            nn.Sigmoid()
        )

        # 4. 空间连续性平滑 (高斯核)
        self.register_buffer('gaussian_kernel', self._get_gaussian_kernel(hidden_dim))

    def _get_gaussian_kernel(self, dim):
        k = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32) / 16.0
        return k.view(1, 1, 3, 3).repeat(dim, 1, 1, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # --- 1. 提取基础空间特征 ---
        feat = self.proj(x)
        feat_dw = self.spatial_dw(feat)

        # --- 2. 坐标感知注入 ---
        # 生成归一化坐标 [B, 2, H, W]
        grid_y, grid_x = torch.meshgrid(torch.linspace(0, 1, H), torch.linspace(0, 1, W), indexing='ij')
        coords = torch.stack([grid_x, grid_y]).unsqueeze(0).repeat(B, 1, 1, 1).to(x.device)
        pos = self.pos_embed(coords)

        # --- 3. 创新：坐标引导的全局光谱校准 ---
        g_spectral = self.global_spectral(feat_dw)  # [B, C, 1, 1]
        g_spectral = g_spectral.expand(-1, -1, H, W)

        # 融合坐标信息和全局信息，生成动态门控
        # 让模型知道在特定的 (x,y) 位置，全局背景应如何影响局部特征
        modulation_gate = self.coord_spectral_gate(torch.cat([g_spectral, pos], dim=1))
        feat_modulated = feat_dw * (1 + modulation_gate)

        # --- 4. 空间连续性校准 (防止滑动窗口产生的潜在断裂) ---
        # 使用组卷积应用高斯平滑
        feat_smooth = F.conv2d(feat_modulated, self.gaussian_kernel, padding=1, groups=C)

        # 最终融合：保留锐利度，同时注入连续性
        out = feat_modulated + 0.1 * feat_smooth
        # out = feat_modulated + x

        # return out, (H, W)
        return out

