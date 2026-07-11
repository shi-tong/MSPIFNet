import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock1D(nn.Module):
    def __init__(self, input_channel, output_channel, kernel=3, stride=1, padding=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=kernel,
            stride=stride,
            padding=padding
        )
        self.bn1 = nn.BatchNorm1d(output_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(
            in_channels=output_channel,
            out_channels=output_channel,
            kernel_size=kernel,
            stride=1,
            padding=padding
        )
        self.bn2 = nn.BatchNorm1d(output_channel)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channel != output_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(input_channel, output_channel, kernel_size=1, stride=stride),
                nn.BatchNorm1d(output_channel)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, layers, in_channels, hidden_channels, out_channels):
        super().__init__()
        
        # Initial layer
        self.initial = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU()
        )
        
        # Feature extraction
        self.layer1 = self._make_layer(block, hidden_channels, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 64, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 128, 256, layers[2], stride=1)
        
        # Final layers
        self.fc = nn.Linear(256, out_channels)

    def _make_layer(self, block, in_channels, out_channels, num_blocks, stride):
        layers = [block(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        B, T, N = x.shape
        x = x.permute(0, 2, 1)  # → [B, N, T]
        
        out = self.initial(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.permute(0,2,1)
        out = self.fc(out)
        
        return F.relu(out) # [B, T, out_channels]

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.feature = None
        self.model = ResNet1D(BasicBlock1D, [2, 2, 2, 2], in_channels=32, hidden_channels=128, out_channels=64)
        self.Init_project = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU()
        )
        self.projection = nn.Linear(64, configs.num_class)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
    def extractor(self):
        return self.feature

    def forward(self, x_enc):
        # project
        x_project = self.Init_project(x_enc)
        # ResNet
        output = self.model(x_project)
        out_pool = self.gap(output.transpose(1, 2)).squeeze(-1)
        # Output
        logits = self.projection(out_pool)  # (batch_size, num_classes)
        # return logits, output
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
            return f'{num / 1e9:.2f}G'
        elif num >= 1e6:
            return f'{num / 1e6:.2f}M'
        elif num >= 1e3:
            return f'{num / 1e3:.2f}K'
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

