import torch
from torch import nn
from torch.nn import functional as F

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


class Quantizer(nn.Module):
    def __init__(self, bit):
        super().__init__()

    def init_from(self, x, *args, **kwargs):
        pass

    def forward(self, x):
        raise NotImplementedError


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


class quan_bn(nn.Module):
    def __init__(self, dim, n_bit):
        super(quan_bn, self).__init__()

        self.bn2 = nn.BatchNorm2d(dim)
        self.bn4 = nn.BatchNorm2d(dim)
        self.bn8 = nn.BatchNorm2d(dim)
        self.bn = None
        self.n_bit = n_bit

    def set_active_filter(self, search_space_sample):
        # print("saf: quanbn")
        self.n_bit = search_space_sample["quantization"][0]

    def get_bit_bn(self, n_bit):
        if n_bit == 2:
            return self.bn2
        elif n_bit == 4:
            return self.bn4
        elif n_bit == 8:
            return self.bn8
        else:
            raise NotImplementedError

    def forward(self, x):
        self.bn = self.get_bit_bn(self.n_bit)
        # print("fwd quan_bn with bit: ", self.n_bit)
        # print(torch.sum(self.bn2.running_mean), torch.sum(self.bn4.running_mean), torch.sum(self.bn8.running_mean))
        return self.bn(x)


class quan_conv2d_1x1(nn.Module):
    def __init__(self, inp, oup, stride, wt_bits, ac_bits):
        super(quan_conv2d_1x1, self).__init__()

        self.quan1 = LsqQuan(ac_bits, True, False, False)
        self.conv1 = HardQuantizeConv(inp, oup, wt_bits, 1, stride, 0)
        self.wt_bits = wt_bits
        self.ac_bits = ac_bits
        self.bn = quan_bn(oup, wt_bits)

    def forward(self, x):
        out = self.quan1(x)
        out = self.conv1(out)
        out = self.bn(out)
        out = F.relu6(out)
        return out


def grad_scale(x, scale):
    y = x
    y_grad = x * scale
    return (y - y_grad).detach() + y_grad


def round_pass(x):
    y = x.round()
    y_grad = x
    return (y - y_grad).detach() + y_grad


class LsqQuan(Quantizer):
    def __init__(self, bit, all_positive=False, symmetric=False, per_channel=True):
        super().__init__(bit)

        if all_positive:
            assert not symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2**bit - 1
        else:
            if symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1)) + 1
                self.thd_pos = 2 ** (bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = -(2 ** (bit - 1))
                self.thd_pos = 2 ** (bit - 1) - 1

        self.per_channel = per_channel
        self.s2 = nn.Parameter(torch.ones(1))
        self.s4 = nn.Parameter(torch.ones(1))
        self.s8 = nn.Parameter(torch.ones(1))
        self.bit = bit
        self.s = None
        self.all_positive = all_positive
        self.symmetric = symmetric
        self.per_channel = per_channel

    def get_bit_scale(self, n_bit):
        if n_bit == 2:
            return self.s2
        elif n_bit == 4:
            return self.s4
        elif n_bit == 8:
            return self.s8
        else:
            raise NotImplementedError

    def set_active_filter(self, search_space_sample):
        # print("saf: lsqquan")
        self.bit = search_space_sample["quantization"][1]
        if self.all_positive:
            assert not self.symmetric, "Positive quantization cannot be symmetric"
            # unsigned activation is quantized to [0, 2^b-1]
            self.thd_neg = 0
            self.thd_pos = 2**self.bit - 1
        else:
            if self.symmetric:
                # signed weight/activation is quantized to [-2^(b-1)+1, 2^(b-1)-1]
                self.thd_neg = -(2 ** (self.bit - 1)) + 1
                self.thd_pos = 2 ** (self.bit - 1) - 1
            else:
                # signed weight/activation is quantized to [-2^(b-1), 2^(b-1)-1]
                self.thd_neg = -(2 ** (self.bit - 1))
                self.thd_pos = 2 ** (self.bit - 1) - 1

    def forward(self, x):
        # print("lsqquan - pos, s, neg", self.thd_pos, self.s, self.thd_neg, sep=os.linesep)
        # print("fwd lsqquan with bit: ", self.bit)
        if self.per_channel:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        else:
            s_grad_scale = 1.0 / ((self.thd_pos * x.numel()) ** 0.5)
        self.s = self.get_bit_scale(self.bit)
        s_scale = grad_scale(self.s, s_grad_scale)
        x = x / s_scale
        x = torch.clamp(x, self.thd_neg, self.thd_pos)
        x = round_pass(x)
        x = x * s_scale
        return x


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class HardQuantizeConv(nn.Module):
    def __init__(
        self, in_chn, out_chn, num_bits, kernel_size=3, stride=1, padding=1, groups=1
    ):
        super(HardQuantizeConv, self).__init__()
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.num_bits = num_bits
        init_act_clip_val = 2.0
        self.clip_val = nn.Parameter(
            torch.Tensor([init_act_clip_val]), requires_grad=False
        )
        self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.shape = (out_chn, in_chn // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(
            (torch.rand(self.shape) - 0.5) * 0.001, requires_grad=True
        )

    def set_active_filter(self, search_space_sample):
        # print("saf: hqc")
        self.num_bits = search_space_sample["quantization"][0]

    def forward(self, x):
        # self.clip_val = self.get_bit_clip(self.num_bits)
        # print("hqc clip vals 2 4 8", self.clip_val, self.clip_val2, self.clip_val4, self.clip_val8, sep=os.linesep)
        # print("fwd hqc with bit: ", self.num_bits)
        real_weights = self.weight
        gamma = (2**self.num_bits - 1) / (2 ** (self.num_bits - 1))
        scaling_factor = gamma * torch.mean(
            torch.mean(
                torch.mean(abs(real_weights), dim=3, keepdim=True), dim=2, keepdim=True
            ),
            dim=1,
            keepdim=True,
        )
        scaling_factor = scaling_factor.detach()
        scaled_weights = real_weights / scaling_factor
        cliped_weights = torch.where(
            scaled_weights < self.clip_val / 2, scaled_weights, self.clip_val / 2
        )
        cliped_weights = torch.where(
            cliped_weights > -self.clip_val / 2, cliped_weights, -self.clip_val / 2
        )
        n = float(2**self.num_bits - 1) / self.clip_val
        quan_weights_no_grad = scaling_factor * (
            torch.round((cliped_weights + self.clip_val / 2) * n) / n
            - self.clip_val / 2
        )
        quan_weights = (
            quan_weights_no_grad.detach() - scaled_weights.detach() + scaled_weights
        )
        y = F.conv2d(
            x,
            quan_weights,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )

        return y


class bottleneck(nn.Module):
    def __init__(self, inp, oup, mid, stride, wt_bits, ac_bits):
        super(bottleneck, self).__init__()

        self.stride = stride
        self.inp = inp
        self.oup = oup

        self.bias11 = LearnableBias(inp)
        self.prelu1 = nn.PReLU(inp)
        self.bias12 = LearnableBias(inp)
        self.quan1 = LsqQuan(ac_bits, True, False, False)
        # self.clip_quan1 = LsqStepSize(torch.tensor(1.0))
        self.conv1 = HardQuantizeConv(inp, mid, wt_bits, 1, 1, 0)
        self.bn1 = quan_bn(mid, wt_bits)

        self.bias21 = LearnableBias(mid)
        self.prelu2 = nn.PReLU(mid)
        self.bias22 = LearnableBias(mid)
        # self.quan2 = LTQ(n_bit)
        self.quan2 = LsqQuan(ac_bits, True, False, False)
        # self.clip_quan2 = LsqStepSize(torch.tensor(1.0))
        self.conv2 = HardQuantizeConv(mid, mid, wt_bits, 3, stride, 1, groups=mid)
        self.bn2 = quan_bn(mid, wt_bits)

        self.bias31 = LearnableBias(mid)
        self.prelu3 = nn.PReLU(mid)
        self.bias32 = LearnableBias(mid)
        # self.quan3 = LTQ(n_bit)
        self.quan3 = LsqQuan(ac_bits, True, False, False)
        # self.clip_quan3 = LsqStepSize(torch.tensor(1.0))
        self.conv3 = HardQuantizeConv(mid, oup, wt_bits, 1, 1, 0)
        self.bn3 = quan_bn(oup, wt_bits)

    def forward(self, x):
        # print("bottlenect fwd")
        # quant_fn = AsymLsqQuantizer
        out = self.bias11(x)
        out = self.prelu1(out)
        out = self.bias12(out)
        out = self.quan1(out)
        # out = quant_fn.apply(out, self.clip_quan1, 4, True)
        out = self.conv1(out)
        out = self.bn1(out)

        out = self.bias21(out)
        out = self.prelu2(out)
        out = self.bias22(out)
        out = self.quan2(out)
        # out = quant_fn.apply(out, self.clip_quan2, 4, True)
        out = self.conv2(out)
        out = self.bn2(out)

        out = self.bias31(out)
        out = self.prelu3(out)
        out = self.bias32(out)
        out = self.quan3(out)
        # out = quant_fn.apply(out, self.clip_quan3, 4, True)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.inp == self.oup and self.stride == 1:
            return out + x

        else:
            return out


class MobileNetV2(nn.Module):
    def __init__(self, wt_bits, ac_bits, input_size=224, num_classes=1000):
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
                        wt_bits,
                        ac_bits,
                    )
                )
            elif i == 18:
                self.feature.append(
                    quan_conv2d_1x1(overall_channel[i - 1], 1280, 1, wt_bits, ac_bits)
                )
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
                            wt_bits,
                            ac_bits,
                        )
                    )
                else:
                    self.feature.append(
                        bottleneck(
                            overall_channel[i - 1],
                            overall_channel[i],
                            mid_channel[i - 1],
                            1,
                            wt_bits,
                            ac_bits,
                        )
                    )

        self.pool1 = nn.AvgPool2d(7)
        self.fc = nn.Linear(1280, 1000)

    def forward(self, x):
        # print("mbv2 fwd")
        for i, block in enumerate(self.feature):
            if i == 0:
                x = block(x)
            elif i == 18:
                x = block(x)
            else:
                x = block(x)

        x = self.pool1(x)
        x = x.view(-1, 1280)
        x = self.fc(x)

        return x
