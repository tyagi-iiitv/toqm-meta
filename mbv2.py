import torch.nn as nn
import torch.nn.functional as F

# stage_out_channel = [44] + [22] + [33] * 2 + [44] * 3 + [88] * 4 + [132] * 3 + [224] * 3 + [448]
stage_out_channel = (
    [32] + [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3 + [320]
)

overall_channel = stage_out_channel

mid_channel = []
for i in range(len(stage_out_channel) - 1):
    if i == 0:
        mid_channel += [stage_out_channel[i]]
    else:
        mid_channel += [6 * stage_out_channel[i]]


class conv2d_3x3(nn.Module):
    def __init__(self, inp, oup, stride):
        super(conv2d_3x3, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)

        return out


class conv2d_1x1(nn.Module):
    def __init__(self, inp, oup, stride):
        super(conv2d_1x1, self).__init__()

        self.conv1 = nn.Conv2d(inp, oup, 1, stride, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)

        return out


class bottleneck(nn.Module):
    def __init__(self, inp, oup, mid, stride):
        super(bottleneck, self).__init__()

        self.stride = stride
        self.inp = inp
        self.oup = oup

        self.conv1 = nn.Conv2d(inp, mid, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)

        self.conv2 = nn.Conv2d(mid, mid, 3, stride, 1, groups=mid, bias=False)
        self.bn2 = nn.BatchNorm2d(mid)

        self.conv3 = nn.Conv2d(mid, oup, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu6(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu6(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.inp == self.oup and self.stride == 1:
            return out + x

        else:
            return out


class MobileNetV2(nn.Module):
    def __init__(self, input_size=224, num_classes=1000):
        super(MobileNetV2, self).__init__()

        self.feature = nn.ModuleList()

        for i in range(19):
            if i == 0:
                self.feature.append(conv2d_3x3(3, overall_channel[i], 2))
            elif i == 1:
                self.feature.append(
                    bottleneck(
                        overall_channel[i - 1],
                        overall_channel[i],
                        mid_channel[i - 1],
                        1,
                    )
                )
            elif i == 18:
                self.feature.append(conv2d_1x1(overall_channel[i - 1], 1280, 1))
            else:
                if (
                    stage_out_channel[i - 1] != stage_out_channel[i]
                    and stage_out_channel[i] != 96
                    and stage_out_channel[i] != 320
                ):
                    self.feature.append(
                        bottleneck(
                            overall_channel[i - 1],
                            overall_channel[i],
                            mid_channel[i - 1],
                            2,
                        )
                    )
                else:
                    self.feature.append(
                        bottleneck(
                            overall_channel[i - 1],
                            overall_channel[i],
                            mid_channel[i - 1],
                            1,
                        )
                    )

        self.pool1 = nn.AvgPool2d(7)
        self.fc = nn.Linear(1280, 1000)

    def forward(self, x):

        for i, block in enumerate(self.feature):
            if i == 0:
                x = block(x)
            elif i == 18:
                x = block(x)
            else:
                x = block(x)

        # print(x.shape, flush=True)
        x = self.pool1(x)
        x = x.view(-1, 1280)
        x = self.fc(x)

        return x
