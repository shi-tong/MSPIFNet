import math
import torch
import torch.nn.functional as F
import torch.nn as nn
class DSConv(nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size, 
                stride=1, 
                padding=0,
                dilation=1, 
                bias=False):
        super(DSConv, self).__init__()
        
        # Depthwise Conv1D (groups=in_channels)
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias
        )
        # Pointwise Conv1D (1×1 Conv)
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,  # 1×1 convolution
            stride=1,
            padding=0,
            bias=bias
        )
        # Optional BatchNorm and Activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, w):
        # input x: (B, T, N)
        x = self.depthwise(w)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x
    
class MCCMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim,
        kernel_size=3,
        dilation=1,
        dropout=0.1,
    ):
        super(MCCMBlock, self).__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv = DSConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation
        )

        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.dropout(out)
        return out

class MCCMStage(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dilations,
        kernel_size=3,
        dropout=0.1,
        reduct_ratio: int = 2
    ):
        super(MCCMStage, self).__init__()
        self.blocks = nn.ModuleList()
        for d in dilations:
            self.blocks.append(
                MCCMBlock(
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    dilation=d,
                    dropout=dropout
                )
            )
        # SCA Mechanism
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim // reduct_ratio)
        self.fc_2 = nn.Linear(hidden_dim // reduct_ratio, hidden_dim)
        self.alpha = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, x, prev_stage_last_output=None):
        current_outputs = []
        for layer, block in enumerate(self.blocks):
            if layer == len(self.blocks) - 1:
                if prev_stage_last_output is not None:
                    
                    if prev_stage_last_output.size(-1) != x.size(-1):
                        prev_stage_last_output = F.interpolate(
                            prev_stage_last_output,
                            size=x.size(-1),
                            mode="linear",
                            align_corners=False
                        )
                    # SCA
                    x_add = x + prev_stage_last_output   # (B, C, T)
                    x_mean = torch.mean(x_add, dim=-1) # B, C
                    x_scale = torch.softmax(x_mean, dim=1) + self.alpha
                    x_squeeze = F.gelu(self.fc_1(x_scale))
                    x_excite = torch.sigmoid(self.fc_2(x_squeeze)).unsqueeze(-1)
                    x = x_add * x_excite
                    
            x = block(x)
            current_outputs.append(x)
        return x, current_outputs[-1]
    
class MCCM(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_dim,
        out_channels: int,
        stage=3,
        dropout=0.1,
        kernel_size=3
    ):
        super(MCCM, self).__init__()
        self.stage = stage
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU()
        )
        self.dilations = self._make_dilations(stage=stage)
        # Create stages
        self.stages = nn.ModuleList()
        for i in range(stage):
            self.stages.append(
                MCCMStage(
                    hidden_dim=hidden_dim,
                    dilations=self.dilations[i],
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )

        # Output projection
        self.output_projs = nn.ModuleList()
        for _ in range(stage):
            self.output_projs.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, out_channels),
                    nn.GELU()
                )
            )

    def _make_kernel_sizes(self, stage, base=3, increments=None):
        if increments is None:
            increments = [6] * stage

        kernel_sizes = []
        for i in range(1, stage + 1):
            sizes = [base + sum(increments[:j]) for j in range(i, 0, -1)]
            kernel_sizes.append(sizes)

        return kernel_sizes

    def _make_dilations(self, stage):
        kernel_sizes = self._make_kernel_sizes(
            stage=stage,
            base=3,
            increments=[0, 8, 16] # 0, 8, 16
        )
        dilations = []
        for ks_list in kernel_sizes:
            dilations.append([
                max(1, (ks - 1) // 2)
                for ks in ks_list
            ])

        return dilations

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(0, 2, 1)
        chain_outputs = []
        prev_stage_last_output = None
        for i in range(self.stage):
            y, last_block_output = self.stages[i](x, prev_stage_last_output)
            prev_stage_last_output = last_block_output
            final_output = y.permute(0, 2, 1)
            final_output = self.output_projs[i](final_output)  # (B, T, out_channels)
            
            chain_outputs.append(final_output)

        return chain_outputs
    
class CIAFM(nn.Module):
    def __init__(
        self,
        in_channels: list,          # [N1, N2, ..., NS]
        hidden_channel: int,
        out_channel: int,
        reduct_ratio: int = 2,
        attn_dropout: float = 0.0,
        return_vis: bool = False,
    ):
        super().__init__()
        assert len(in_channels) >= 1, "num_channels (S) must be >= 1"
        self.S = len(in_channels)
        self.C = max(1, hidden_channel // max(1, reduct_ratio))  # sub_channel
        self.out_channel = out_channel
        self.return_vis = return_vis
        self.scale = 1.0 / math.sqrt(self.C)
        # Per-scale input projection
        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Linear(Ni, self.C),
                nn.GELU(),
            ) for Ni in in_channels
        ])

        # Shared K, V_base from concatenated projected features
        self.K_proj = nn.Linear(self.C * self.S, self.C)
        self.V_proj = nn.Linear(self.C * self.S, self.C)

        # Per-scale Q
        self.Q_projs = nn.ModuleList([nn.Linear(self.C, self.C) for _ in range(self.S)])

        # Gated shared V per-scale
        gate_hidden = max(1, self.C // 4)
        self.gates = nn.ModuleList()
        for _ in range(self.S):
            # outputs g_i (C)
            self.gates.append(nn.Sequential(
                nn.Linear(self.C, gate_hidden),
                nn.GELU(),
                nn.Linear(gate_hidden, self.C)
            ))

        self.attn_dropout = nn.Dropout(attn_dropout) if attn_dropout > 0 else nn.Identity()
        self.fuse_score = nn.Linear(self.C, 1, bias=False)
        self.out_proj = nn.Conv1d(self.C, self.out_channel, kernel_size=1)
        self.pre_norm  = nn.LayerNorm(self.C)   # on per-scale projected inputs
        self.head_norm = nn.LayerNorm(self.C)   # on per-scale attended outputs
        self.post_norm = nn.LayerNorm(self.out_channel)
        self._init_weights()

    def _init_weights(self):
        # Make fuse_score small to start with near-uniform scale fusion
        nn.init.normal_(self.fuse_score.weight, std=1e-4)

    def forward(self, x: list):
        assert len(x) == self.S
        B, T = x[0].shape[0], x[0].shape[1]
        x_proj = []
        for i in range(self.S):
            xi = self.input_proj[i](x[i])
            xi = self.pre_norm(xi)
            x_proj.append(xi)

        # Concatenate and build shared K and V_base
        x_cat = torch.cat(x_proj, dim=-1)
        K = self.K_proj(x_cat)
        V_base = self.V_proj(x_cat)

        # Build per-scale gated V_i
        conds = [xi.mean(dim=1) for xi in x_proj]  # list of (B,C)
        V_list = []
        for i in range(self.S):
            gate_vec = self.gates[i](conds[i])
            g = torch.sigmoid(gate_vec).unsqueeze(1)
            V_i = g * V_base
            V_list.append(V_i)

        attn_weights_list = [] if self.return_vis else None
        Y_list = []
        for i in range(self.S):
            Q = self.Q_projs[i](x_proj[i])
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            A = F.softmax(scores, dim=-1)
            A = self.attn_dropout(A)
            Y_i = torch.matmul(A, V_list[i])
            Y_i = self.head_norm(Y_i)
            Y_list.append(Y_i)
            if self.return_vis:
                attn_weights_list.append(A.detach())
        Y_stack = torch.stack(Y_list, dim=-1)
        logits = self.fuse_score(Y_stack.permute(0, 1, 3, 2))
        alpha = F.softmax(logits, dim=2)                       # (B,T,S,1)
        alpha = alpha.permute(0, 1, 3, 2)                      # (B,T,1,S)
        Y_fused = (Y_stack * alpha).sum(dim=-1)

        y = self.out_proj(Y_fused.permute(0, 2, 1))
        y = y.permute(0, 2, 1)
        y = self.post_norm(y)

        if self.return_vis:
            # alpha for cross-scale weights visualization
            alpha_vis = alpha.squeeze(2).detach()
            return y, alpha_vis
        else:
            return y, None 
        
class MSPIFNet(nn.Module):
    def __init__(self, in_channel, hidden_channel, sub_channel, out_channel, stage=3):
        super().__init__()
        self.stage = stage
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.sub_channel = sub_channel
        self.out_channel = out_channel
        self.MCCM = MCCM(
            in_channels=self.in_channel,
            hidden_dim=self.hidden_channel,
            out_channels=self.sub_channel,
            stage=self.stage,
            dropout=0.1,
            kernel_size=3,
        )
        self.CIAFM = CIAFM(
            [self.sub_channel] * stage,
            128,
            self.sub_channel,
            return_vis=True,
            reduct_ratio=2
        )
        self.Map = nn.Sequential(
            nn.Linear(self.sub_channel, self.out_channel),
            nn.GELU()
        )
        self.norm = nn.LayerNorm(self.out_channel)
        self.residual = nn.Conv1d(self.in_channel, self.out_channel, 1)

    def forward(self, x):
        mccm_features = self.MCCM(x)
        cfa_attention, atten_outputs = self.CIAFM(mccm_features)
        out_features = self.Map(cfa_attention)
        residual_x = self.residual(x.permute(0, 2, 1))
        if residual_x.size(-1) != out_features.size(1):
            residual_x = F.adaptive_avg_pool1d(
                residual_x,
                out_features.size(1)
            )
        residual_x = residual_x.permute(0, 2, 1)
        out_norm = self.norm(out_features + residual_x)
        return out_norm, atten_outputs

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.Init_project = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU()
        )
        self.model = MSPIFNet(
            in_channel=32,
            hidden_channel=128,
            sub_channel=64,
            out_channel=64
        )
        self.projection = nn.Linear(64, configs.num_class)

    def forward(self, x_enc):
        x_project = self.Init_project(x_enc)
        output, _ = self.model(x_project)
        output_mean = output.mean(dim=1)
        logits = self.projection(output_mean)
        return logits

if __name__ == '__main__':
    import torch
    from thop import profile

    class Configs:
        def __init__(self):
            self.input_dim = 1
            self.seq_len = 512
            self.num_class = 8

    def format_num(num):
        if num >= 1e9:
            return f'{num / 1e9:.3f}G'
        elif num >= 1e6:
            return f'{num / 1e6:.3f}M'
        elif num >= 1e3:
            return f'{num / 1e3:.3f}K'
        else:
            return str(num)
    input_z = torch.randn((8, 512, 1))
    config = Configs()
    model = Model(configs=config)
    model.eval()
    with torch.no_grad():
        out = model(input_z)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, _ = profile(model, inputs=(input_z,), verbose=False)
    flops = 2 * macs
    print(f"Trainable parameters: {format_num(params)}")
    print(f"FLOPs: {format_num(flops)}")
