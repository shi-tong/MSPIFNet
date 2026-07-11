import torch
import torch.nn as nn

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PyConv4(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7, 9], stride=1, pyconv_groups=[1, 4, 8, 16]):
        super(PyConv4, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])
        self.conv2_4 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[3], padding=pyconv_kernels[3] // 2,
                            stride=stride, groups=pyconv_groups[3])

    def forward(self, x):
        out1 = self.conv2_1(x)
        out2 = self.conv2_2(x)
        out3 = self.conv2_3(x)
        out4 = self.conv2_4(x)
        out = out1 + out2 + out3 + out4
        return out


class PyConv3(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5, 7], stride=1, pyconv_groups=[1, 4, 8]):
        super(PyConv3, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])
        self.conv2_3 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[2], padding=pyconv_kernels[2] // 2,
                            stride=stride, groups=pyconv_groups[2])

    def forward(self, x):
        out1 = self.conv2_1(x)
        out2 = self.conv2_2(x)
        out3 = self.conv2_3(x)
        out = out1 + out2 + out3
        return out


class PyConv2(nn.Module):

    def __init__(self, inplans, planes, pyconv_kernels=[3, 5], stride=1, pyconv_groups=[1, 4]):
        super(PyConv2, self).__init__()
        self.conv2_1 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[0], padding=pyconv_kernels[0] // 2,
                            stride=stride, groups=pyconv_groups[0])
        self.conv2_2 = conv(inplans, planes // 4, kernel_size=pyconv_kernels[1], padding=pyconv_kernels[1] // 2,
                            stride=stride, groups=pyconv_groups[1])

    def forward(self, x):
        out1 = self.conv2_1(x)
        out2 = self.conv2_2(x)
        out = out1 + out2
        return out
        # return torch.cat((self.conv2_1(x), self.conv2_2(x)), dim=1)


def mlt_dwconv(inplans, planes, pyconv_kernels, stride=1, pyconv_groups=[1]):
    if len(pyconv_kernels) == 1:
        return conv(inplans, planes, kernel_size=pyconv_kernels[0], stride=stride, groups=pyconv_groups[0])
    elif len(pyconv_kernels) == 2:
        return PyConv2(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 3:
        return PyConv3(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)
    elif len(pyconv_kernels) == 4:
        return PyConv4(inplans, planes, pyconv_kernels=pyconv_kernels, stride=stride, pyconv_groups=pyconv_groups)


class MSCL(nn.Module):
    expansion = 4
    def __init__(self, num_channel, pyconv_kernels, pyconv_groups, padding=0,
                 dilation=1):  # 未使用空洞 gsu_dilation=1
        super(MSCL, self).__init__()
        self.conv1 = conv1x1(num_channel, num_channel//2)
        self.selu = nn.SELU(inplace=True)
        self.conv2 = mlt_dwconv(num_channel//2, num_channel//2, pyconv_kernels=pyconv_kernels, stride=1,
                                pyconv_groups=pyconv_groups)
        self.conv3 = conv1x1(num_channel//8, num_channel)

        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='selu')


    def forward(self, x):
        out = self.conv1(x)
        out = self.selu(out)
        out = self.conv2(out)
        out = self.selu(out)
        out = self.conv3(out)
        attn_map = torch.sigmoid(out)
        return x * attn_map


class MSCRB(nn.Module):
    def __init__(self, in_channel, out_channel, pyconv_kernels, pyconv_groups, stride):
        super(MSCRB, self).__init__()
        if stride != 1 or in_channel != out_channel:
            self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False)
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='selu')
            self.bn1 = nn.BatchNorm1d(out_channel)
            nn.init.constant_(self.bn1.weight, 1)
            nn.init.constant_(self.bn1.bias, 0)
            self.downsample = nn.Sequential(self.conv1, self.bn1)
        else:
            self.downsample = None

        self.conv2 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='selu')
        self.bn2 = nn.BatchNorm1d(out_channel)
        nn.init.constant_(self.bn2.weight, 1)
        nn.init.constant_(self.bn2.bias, 0)

        self.conv3 = nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv3.weight, mode='fan_out', nonlinearity='selu')
        self.bn3 = nn.BatchNorm1d(out_channel)
        nn.init.constant_(self.bn3.weight, 1)
        nn.init.constant_(self.bn3.bias, 0)

        self.mscl = MSCL(in_channel, pyconv_kernels, pyconv_groups)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.mscl(x)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.selu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.selu(out)

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.input_dim = 1
        self.seq_len = configs.seq_len
        self.num_class = configs.num_class
        self.inplanes = 32  # or 32
        self.MSCRB_num_layers = 4 # Layers must match the below lists
        self.pyconv_kernel_list = [
            [7, 15, 31, 63], 
            [7, 15, 31, 63],
            [7, 15, 31], 
            [7, 15]
        ]
        self.pyconv_group_list = [
            [2, 2, 2, 2], 
            [2, 2, 2, 2],
            [2, 2, 2], 
            [2, 2]
        ]

        self.conv = nn.Conv1d(
            self.input_dim, self.inplanes, kernel_size=3,
            stride=2, padding=1, bias=False)  # 512
        self.bn = nn.BatchNorm1d(num_features=self.inplanes)
        self.selu = nn.SELU(inplace=True)

        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='selu')
        nn.init.constant_(self.bn.weight, 0.5)
        nn.init.constant_(self.bn.bias, 0)

        self.MSCRB_list = nn.ModuleList()
        for i in range(self.MSCRB_num_layers // 2):
            self.MSCRB_list.append(
                MSCRB(
                    self.inplanes * 2 ** i, self.inplanes * 2 ** i,
                    self.pyconv_kernel_list[i*2], self.pyconv_group_list[i*2], 1
                )
            )
            self.MSCRB_list.append(
                MSCRB(
                    self.inplanes * 2 ** i, self.inplanes * 2 ** (i + 1),
                    self.pyconv_kernel_list[i*2 + 1], self.pyconv_group_list[i*2 + 1], 1
                )
            )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.inplanes * self.MSCRB_num_layers, self.num_class)

    def forward(self, x):
        # x: B, T, N
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.bn(x)
        x = self.selu(x)

        for _, MSCRB_layer in enumerate(self.MSCRB_list):
            x = MSCRB_layer(x)

        x_pool = self.gap(x)

        x_pool = x_pool.view(x_pool.size(0), -1)
        logits = self.fc(x_pool)

        # return logits, x
        return logits
'''
Paper name: Multiscale calibration networks with pseudo label for bearing fault diagnosis under class-imbalanced data and multi-rate sampling scenarios
doi: https://doi.org/10.1016/j.aei.2025.103694
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
    print(f"Output shape: {out.shape}")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    macs, _ = profile(model, inputs=(input_z,), verbose=False)
    flops = 2 * macs
    print(f"Trainable parameters: {format_num(params)}")
    print(f"FLOPs: {format_num(flops)}")