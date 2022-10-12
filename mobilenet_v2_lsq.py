import math

import torch
from fblearner.flow.projects.users.anjultyagi.utils import LsqQuan
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


class quan_conv2d_1x1(nn.Module):
    def __init__(self, inp, oup, stride, n_bit):
        super(quan_conv2d_1x1, self).__init__()

        self.quan1 = PACT(n_bit)
        self.conv1 = HardQuantizeConv(inp, oup, n_bit, 1, stride, 0)
        self.bn1 = nn.BatchNorm2d(oup)

    def forward(self, x):

        out = self.quan1(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu6(out)

        return out


class LearnableBias(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(torch.zeros(1, out_chn, 1, 1), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class PACT(nn.Module):
    def __init__(self, num_bits, init_act_clip_val=2):
        super(PACT, self).__init__()
        self.num_bits = num_bits
        self.clip_val = nn.Parameter(
            torch.Tensor([init_act_clip_val]), requires_grad=True
        )

    def forward(self, x):
        x = F.relu(x)
        x = torch.where(x < self.clip_val, x, self.clip_val)
        n = float(2**self.num_bits - 1) / self.clip_val
        x_forward = torch.round(x * n) / n
        out = x_forward + x - x.detach()
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
            torch.Tensor([init_act_clip_val]), requires_grad=True
        )
        self.zero = nn.Parameter(torch.Tensor([0]), requires_grad=False)
        self.shape = (out_chn, in_chn // groups, kernel_size, kernel_size)
        self.weight = nn.Parameter(
            (torch.rand(self.shape) - 0.5) * 0.001, requires_grad=True
        )

    def forward(self, x):

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


class LTQ(nn.Module):
    def __init__(self, num_bits):
        super(LTQ, self).__init__()
        init_range = 2.0
        self.n_val = 2**num_bits - 1
        self.interval = init_range / self.n_val
        self.start = nn.Parameter(torch.Tensor([0.0]), requires_grad=True)
        self.a = nn.Parameter(
            torch.Tensor([self.interval] * self.n_val), requires_grad=True
        )
        self.scale1 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)
        self.scale2 = nn.Parameter(torch.Tensor([1.0]), requires_grad=True)

        self.two = nn.Parameter(torch.Tensor([2.0]), requires_grad=False)
        self.one = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
        self.zero = nn.Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.minusone = nn.Parameter(torch.Tensor([-1.0]), requires_grad=False)
        self.eps = nn.Parameter(torch.Tensor([1e-3]), requires_grad=False)

    def forward(self, x):

        x = x * self.scale1

        x_forward = x
        x_backward = x
        step_right = self.zero + 0.0

        a_pos = torch.where(self.a > self.eps, self.a, self.eps)

        for i in range(self.n_val):
            step_right += self.interval
            if i == 0:
                thre_forward = self.start + a_pos[0] / 2
                thre_backward = self.start + 0.0
                x_forward = torch.where(x > thre_forward, step_right, self.zero)
                x_backward = torch.where(
                    x > thre_backward,
                    self.interval / a_pos[i] * (x - thre_backward)
                    + step_right
                    - self.interval,
                    self.zero,
                )
            else:
                thre_forward += a_pos[i - 1] / 2 + a_pos[i] / 2
                thre_backward += a_pos[i - 1]
                x_forward = torch.where(x > thre_forward, step_right, x_forward)
                x_backward = torch.where(
                    x > thre_backward,
                    self.interval / a_pos[i] * (x - thre_backward)
                    + step_right
                    - self.interval,
                    x_backward,
                )

        thre_backward += a_pos[i]
        x_backward = torch.where(x > thre_backward, self.two, x_backward)

        out = x_forward.detach() + x_backward - x_backward.detach()
        out = out * self.scale2

        return out


class AsymLsqQuantizer(torch.autograd.Function):
    """
    Asymetric LSQ quantization. Modified from LSQ.
    """

    @staticmethod
    def forward(ctx, input, alpha, num_bits, layerwise):
        """
        :param input: input to be quantized
        :param alpha: the step size
        :param num_bits: quantization bits
        :param layerwise: rowwise quant
        :return: quantized output
        """
        if not layerwise:
            # TODO
            raise NotImplementedError
        ctx.num_bits = num_bits
        if num_bits == 32:
            return input

        Qn = 0
        Qp = 2 ** (num_bits) - 1
        # asymmetric: make sure input \in [0, +\inf], remember to add it back
        min_val = input.min().item()
        input_ = input - min_val

        assert alpha > 0, "alpha = {:.6f} becomes non-positive".format(alpha)
        if alpha.item() == 1.0 and (not alpha.initialized):
            alpha.initialize_wrapper(
                input, num_bits, symmetric=False, init_method="uniform"
            )
            # Qp = 2 ** (num_bits) - 1
            # init_val = 4 * input.abs().mean() / math.sqrt(Qp)
            # alpha.data.copy_(init_val)

        grad_scale = 1.0 / math.sqrt(input.numel() * Qp)
        # grad_scale = 1.0
        ctx.save_for_backward(input_, alpha)
        ctx.other = grad_scale, Qn, Qp
        q_w = (input_ / alpha).round().clamp(Qn, Qp)
        w_q = q_w * alpha
        w_q = w_q + min_val
        return w_q

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.num_bits == 32:
            return grad_output, None, None, None

        input_, alpha = ctx.saved_tensors
        grad_scale, Qn, Qp = ctx.other
        q_w = input_ / alpha
        indicate_small = (q_w < Qn).float()
        indicate_big = (q_w > Qp).float()
        indicate_middle = (
            1.0 - indicate_small - indicate_big
        )  # this is more cpu-friendly than torch.ones(input_.shape)
        grad_alpha = (
            (
                (
                    indicate_small * Qn
                    + indicate_big * Qp
                    + indicate_middle * (-q_w + q_w.round())
                )
                * grad_output
                * grad_scale
            )
            .sum()
            .unsqueeze(dim=0)
        )
        grad_input = indicate_middle * grad_output
        return grad_input, grad_alpha, None, None


class LsqStepSize(nn.Parameter):
    def __init__(self, tensor):
        super(LsqStepSize, self).__new__(nn.Parameter, data=tensor)
        self.initialized = False

    def _initialize(self, init_tensor):
        assert not self.initialized, "already initialized."
        self.data.copy_(init_tensor)
        # print('Stepsize initialized to %.6f' % self.data.item())
        self.initialized = True

    def initialize_wrapper(self, tensor, num_bits, symmetric, init_method="default"):
        # input: everthing needed to initialize step_size
        Qp = 2 ** (num_bits - 1) - 1 if symmetric else 2 ** (num_bits) - 1
        if init_method == "default":
            init_val = (
                2 * tensor.abs().mean() / math.sqrt(Qp)
                if symmetric
                else 4 * tensor.abs().mean() / math.sqrt(Qp)
            )
        elif init_method == "uniform":
            init_val = 1.0 / (2 * Qp + 1) if symmetric else 1.0 / Qp

        self._initialize(init_val)


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
        self.bn1 = nn.BatchNorm2d(mid)

        self.bias21 = LearnableBias(mid)
        self.prelu2 = nn.PReLU(mid)
        self.bias22 = LearnableBias(mid)
        # self.quan2 = LTQ(n_bit)
        self.quan2 = LsqQuan(ac_bits, True, False, False)
        # self.clip_quan2 = LsqStepSize(torch.tensor(1.0))
        self.conv2 = HardQuantizeConv(mid, mid, wt_bits, 3, stride, 1, groups=mid)
        self.bn2 = nn.BatchNorm2d(mid)

        self.bias31 = LearnableBias(mid)
        self.prelu3 = nn.PReLU(mid)
        self.bias32 = LearnableBias(mid)
        # self.quan3 = LTQ(n_bit)
        self.quan3 = LsqQuan(ac_bits, True, False, False)
        # self.clip_quan3 = LsqStepSize(torch.tensor(1.0))
        self.conv3 = HardQuantizeConv(mid, oup, wt_bits, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(oup)

    def forward(self, x):

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
                    quan_conv2d_1x1(overall_channel[i - 1], 1280, 1, wt_bits)
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
