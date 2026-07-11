import torch
import torch.nn as nn

class PSCA_Module(nn.Module):
    def __init__(self, in_channels, out_channels, G, stride):
        super(PSCA_Module, self).__init__()
        self.G = G
        self.stride = stride
        self.conv3x3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv5x5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, stride=stride, padding=2)
        self.conv7x7 = nn.Conv1d(in_channels, out_channels, kernel_size=7, stride=stride, padding=3)
        self.conv9x9 = nn.Conv1d(in_channels, out_channels, kernel_size=9, stride=stride, padding=4)

    def forward(self, x):
        x1 = self.conv3x3(x)
        x2 = self.conv5x5(x)
        x3 = self.conv7x7(x)
        x4 = self.conv9x9(x)
        return torch.cat([x1, x2, x3, x4], dim=1)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_dim = 1
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.PARConv_num_layers = 4
        
        # Convolution block
        self.conv1 = nn.Conv1d(self.input_dim, 64, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # PAR block 1
        self.par1_conv1 = nn.Conv1d(64, 64, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU()
        self.psca1 = PSCA_Module(64, 16, G=16, stride=2)

        self.par1_conv2 = nn.Conv1d(64, 128, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU()

        # PAR block 2
        self.par2_conv1 = nn.Conv1d(128, 128, kernel_size=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.relu4 = nn.ReLU()
        self.psca2 = PSCA_Module(128, 32, G=16, stride=2)

        self.par2_conv2 = nn.Conv1d(128, 256, kernel_size=1)
        self.bn5 = nn.BatchNorm1d(256)
        self.relu5 = nn.ReLU()

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(256, self.num_class)

    def forward(self, x):
        # x: B, T, N
        # Convolution block
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        # PAR block 1
        x = self.par1_conv1(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.psca1(x)

        x = self.par1_conv2(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # PAR block 2
        x = self.par2_conv1(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.psca2(x)

        x = self.par2_conv2(x)
        x = self.bn5(x)
        x = self.relu5(x)

        x_pool = self.gap(x)

        x_flatten = torch.flatten(x_pool, 1)
        logits = self.fc(x_flatten)

        # return logits, x
        return logits
'''
Paper name: Lightweight pyramid attention residual network for intelligent fault diagnosis of machine under sharp speed variation
DOI link: https://doi.org/10.1016/j.ymssp.2024.111824
'''
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