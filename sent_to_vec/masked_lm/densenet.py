"""
DenseNet architecture
"""

from math import sqrt
import torch
import torch.nn as nn

import time

class Transition(nn.Sequential):
    """
    Transiton btw dense blocks:
    BN > ReLU > Conv(k=1) to reduce the number of channels
    """
    def __init__(self, num_input_features, num_output_features, init_weights=0):
        super(Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        conv = nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                bias=False)
        if init_weights == "manual":
            std = sqrt(2 / num_input_features)
            conv.weight.data.normal_(0, std)
        self.add_module('conv', conv)

    def forward(self, x, *args):
        return super(Transition, self).forward(x)


class Transition2(nn.Sequential):
    """
    Transiton btw dense blocks:
    ReLU > Conv(k=1) to reduce the number of channels
    """
    def __init__(self, num_input_features, num_output_features):
        super(Transition2, self).__init__()
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                bias=False))
        
    def forward(self, x, *args):
        return super(Transition2, self).forward(x)


class AsymmetricMaskedConv2d(nn.Conv2d):
    """
    Masked (autoregressive) conv2d kx1 kernel
    FIXME: particular case of the MaskedConv2d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 groups=1, bias=False):
        pad = (dilation * (kernel_size - 1)) // 2
        super().__init__(in_channels, out_channels,
                         (kernel_size, 1),
                         padding=(pad, 0),
                         groups=groups,
                         dilation=dilation,
                         bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:, :] = 0
        self.incremental_state = torch.zeros(1, 1, 1, 1)

    def forward(self, x, *args):
        self.weight.data *= self.mask
        return super().forward(x)

    def update(self, x):
        k = self.weight.size(2) // 2 + 1
        buffer = self.incremental_state
        if buffer.size(2) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            # shift the buffer and add the recent input:
            buffer[:, :, :-1, :] = buffer[:, :, 1:, :].clone()
            buffer[:, :, -1:, :] = x[:, :, -1:, :]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class MaskedConv2d(nn.Conv2d):
    """
    Masked (autoregressive) conv2d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 groups=1, bias=False):
        pad = (dilation * (kernel_size - 1)) // 2
        super(MaskedConv2d, self).__init__(in_channels, out_channels,
                                           kernel_size,
                                           padding=pad,
                                           groups=groups,
                                           dilation=dilation,
                                           bias=bias)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        if kH > 1:
            self.mask[:, :, kH // 2 + 1:, :] = 0
        self.incremental_state = torch.zeros(1, 1, 1, 1)

    def forward(self, x, *args):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)

    def update(self, x):
        k = self.weight.size(2) // 2 + 1
        buffer = self.incremental_state
        if buffer.size(2) < k:
            output = self.forward(x)
            self.incremental_state = x.clone()
        else:
            # shift the buffer and add the recent input:
            buffer[:, :, :-1, :] = buffer[:, :, 1:, :].clone()
            buffer[:, :, -1:, :] = x[:, :, -1:, :]
            output = self.forward(buffer)
            self.incremental_state = buffer.clone()
        return output


class GatedConv2d(MaskedConv2d):
    """
    Gated version of the masked conv2d
    """
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, dilation=1,
                 bias=False, groups=1):
        super(GatedConv2d, self).__init__(in_channels,
                                          2*out_channels,
                                          kernel_size,
                                          dilation=dilation,
                                          bias=bias)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = super(GatedConv2d, self).forward(x)
        mask, out = x.chunk(2, dim=1)
        mask = self.sigmoid(mask)
        return out * mask


def _setup_conv_dilated(num_input_features, kernel_size, params, first=False):
    """
    Common setup of convolutional layers in a dense layer
    """
    bn_size = params.get('bn_size', 4)
    growth_rate = params.get('growth_rate', 32)
    bias = params.get('bias', 0)
    drop_rate = params.get('conv_dropout', 0.)
    init_weights = params.get('init_weights', 0)
    weight_norm = params.get('weight_norm', 0)
    gated = params.get('gated', 0)
    dilation = params.get('dilation', 2)
    print('Dilation: ', dilation)

    CV = GatedConv2d if gated else MaskedConv2d
    interm_features = bn_size * growth_rate
    conv1 = nn.Conv2d(
        num_input_features,
        interm_features,
        kernel_size=1,
        bias=bias)
    conv2 = CV(
        interm_features,
        interm_features,
        kernel_size=kernel_size,
        bias=bias)

    conv3 = CV(
        interm_features,
        growth_rate,
        kernel_size=kernel_size,
        bias=bias,
        dilation=dilation)

    if init_weights == "manual":
        if not first:
            # proceeded by dropout and relu
            cst = 2 * (1 - drop_rate) 
        else:
            cst = 1
        # n_l = num_input_features 
        std1 = sqrt(cst / num_input_features)
        conv1.weight.data.normal_(0, std1)
        # n_l = num_input_features * k * [(k-1)/2]
        # only relu
        std2 = sqrt(2 / (interm_featires * kernel_size *
                         (kernel_size - 1) // 2))
        conv2.weight.data.normal_(0, std2)
        conv3.weight.data.normal_(0, std2)

        if bias:
            conv1.bias.data.zero_()
            conv2.bias.data.zero_()
            conv3.bias.data.zero_()

    elif init_weights == "kaiming":
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity='relu')
        nn.init.kaiming_normal_(conv2.weight, mode="fan_out", nonlinearity='relu')
        nn.init.kaiming_normal_(conv3.weight, mode="fan_out", nonlinearity='relu')

    if weight_norm:
        conv1 = nn.utils.weight_norm(conv1, dim=0) # dim = None ?
        conv2 = nn.utils.weight_norm(conv2, dim=0)
        conv3 = nn.utils.weight_norm(conv3, dim=0)

    return conv1, conv2, conv3


def _setup_conv(num_input_features, kernel_size, params, first=False):
    """
    Common setup of convolutional layers in a dense layer
    """
    bn_size = params.get('bn_size', 4)
    growth_rate = params.get('growth_rate', 32)
    bias = params.get('bias', 0)
    drop_rate = params.get('conv_dropout', 0.)
    init_weights = params.get('init_weights', 0)
    weight_norm = params.get('weight_norm', 0)
    gated = params.get('gated', 0)
    depthwise = params.get('depthwise', 0)

    CV = GatedConv2d if gated else MaskedConv2d
    interm_features = bn_size * growth_rate
    conv1 = nn.Conv2d(
        num_input_features,
        interm_features,
        kernel_size=1,
        bias=bias)
    gp = growth_rate if depthwise else 1
    conv2 = CV(
        interm_features,
        growth_rate,
        kernel_size=kernel_size,
        bias=bias,
        groups=gp)

    if init_weights == "manual":
        # Init weights so that var(in) = var(out)
        if not first:
            # proceeded by dropout and relu
            cst = 2 * (1 - drop_rate) 
        else:
            cst = 1
        # n_l = num_input_features 
        std1 = sqrt(cst / num_input_features)
        conv1.weight.data.normal_(0, std1)
        # n_l = num_input_features * k * [(k-1)/2]
        # only relu
        std2 = sqrt(2 / (bn_size * growth_rate * kernel_size *
                                   (kernel_size - 1) // 2))
        conv2.weight.data.normal_(0, std2)
        if bias:
            conv1.bias.data.zero_()
            conv2.bias.data.zero_()

    elif init_weights == "kaiming":
        #  Use pytorch's kaiming_normal_
        nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity='relu')
        nn.init.kaiming_normal_(conv2.weight, mode="fan_out", nonlinearity='relu')

    if weight_norm:
        conv1 = nn.utils.weight_norm(conv1, dim=0) # dim = None ?
        conv2 = nn.utils.weight_norm(conv2, dim=0)

    return conv1, conv2


class _MainDenseLayer(nn.Module):
    """
    Main dense layer declined in 2 variants
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params
                ):
        super().__init__()
        self.kernel_size = kernel_size
        self.bn_size = params.get('bn_size', 4)
        self.growth_rate = params.get('growth_rate', 32)
        self.drop_rate = params.get('conv_dropout', 0.)
        
    def forward(self, x):
        new_features = self.seq(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    def reset_buffers(self):
        for layer in list(self.seq.children()):
            if isinstance(layer, MaskedConv2d):
                layer.incremental_state = torch.zeros(1, 1, 1, 1)
        # self.conv2.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        for layer in list(self.seq.children()):
            if isinstance(layer, MaskedConv2d):
                x = layer.update(x)
            else:
                x = layer(x)
        return torch.cat([res, x], 1)

    def track(self, x):
        new_features = self.seq(x)
        return x, new_features


class DenseLayer(_MainDenseLayer):
    """
    BN > ReLU > Conv(1) > BN > ReLU > Conv(k)
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params)
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            conv1,
            nn.BatchNorm2d(self.bn_size * self.growth_rate),
            nn.ReLU(inplace=True),
            conv2
            )


class DenseLayer_midDP(_MainDenseLayer):
    """
    BN > ReLU > Conv(1) > Dropout > BN > ReLU > Conv(k)
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params)
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            conv1,
            nn.Dropout(p=self.drop_rate, inplace=True),
            nn.BatchNorm2d(self.bn_size * self.growth_rate),
            nn.ReLU(inplace=True),
            conv2
            )


class DenseLayer_noBN(_MainDenseLayer):
    """
    ReLU > Conv(1) > ReLU > Conv(k)
    #TODO: check activ' var
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2 = _setup_conv(num_input_features, kernel_size, params, first=first)
        self.seq = nn.Sequential(
            nn.ReLU(inplace=True),
            conv1,
            nn.ReLU(inplace=True),
            conv2
        )


class DenseLayer_Dil(_MainDenseLayer):
    """
    BN > ReLU > Conv(1)
    > BN > ReLU > Conv(k)
    > BN > ReLU > Conv(k, dilated)
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__(num_input_features, kernel_size, params)
        conv1, conv2, conv3 = _setup_conv_dilated(num_input_features,
                                                  kernel_size,
                                                  params)
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            conv1,
            nn.BatchNorm2d(self.bn_size * self.growth_rate),
            nn.ReLU(inplace=True),
            conv2,
            nn.BatchNorm2d(self.bn_size * self.growth_rate),
            nn.ReLU(inplace=True),
            conv3
            )



class DenseLayer_Asym(nn.Module):
    """
    Dense layer with asymmetric convolution ie decompose a 3x3 conv into
    a 3x1 1D conv followed by a 1x3 1D conv.
    As suggested in: 
    Efficient Dense Modules of Asymmetric Convolution for
    Real-Time Semantic Segmentation
    https://arxiv.org/abs/1809.06323
    """
    def __init__(self,
                 num_input_features,
                 kernel_size,
                 params,
                 first=False
                ):
        super().__init__()
        self.kernel_size = kernel_size
        self.drop_rate = params.get('conv_dropout', 0.)
        bias = params.get('bias', 0)
        bn_size = params.get('bn_size', 4)
        growth_rate = params.get('growth_rate', 32)
        dim1 = bn_size * growth_rate
        dim2 = bn_size // 2 * growth_rate

        conv1 = nn.Conv2d(
            num_input_features,
            dim1,
            kernel_size=1,
            bias=False)

        pad = (kernel_size - 1) // 2
        conv2s = nn.Conv2d(
            dim1,
            dim2,
            kernel_size=(1, kernel_size),
            padding=(0, pad),
            bias=False)

        conv2t = AsymmetricMaskedConv2d(
            dim2,
            growth_rate,
            kernel_size=kernel_size,
            bias=False)
        
        self.seq = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            conv1,
            nn.BatchNorm2d(dim1),
            nn.ReLU(inplace=True),
            conv2s,
            conv2t
            )

    def forward(self, x):
        new_features = self.seq(x)
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)

    def reset_buffers(self):
        for layer in list(self.seq.children()):
            if isinstance(layer, AsymmetricMaskedConv2d):
                layer.incremental_state = torch.zeros(1, 1, 1, 1)
        # self.conv2.incremental_state = torch.zeros(1, 1, 1, 1)

    def update(self, x):
        maxh = self.kernel_size // 2 + 1
        if x.size(2) > maxh:
            x = x[:, :, -maxh:, :].contiguous()
        res = x
        for layer in list(self.seq.children()):
            if isinstance(layer, AsymmetricMaskedConv2d):
                x = layer.update(x)
            else:
                x = layer(x)
        return torch.cat([res, x], 1)

    def track(self, x):
        new_features = self.seq(x)
        return x, new_features


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers,
                 num_input_features,
                 kernels,
                 params):
        super(DenseBlock, self).__init__()
        layer_type = params.get('layer_type', 1)
        growth_rate = params.get('growth_rate', 32)
        if layer_type == "regular":
            LayerModule = DenseLayer
        elif layer_type == "mid-dropout":  # Works fine, basically another dropout
            LayerModule = DenseLayer_midDP
        elif layer_type == "nobn":  # W/o BN works fine if weights initialized "correctly"
            LayerModule = DenseLayer_noBN
        elif layer_type == "asym":
            LayerModule = DenseLayer_Asym
        elif layer_type == "dilated": # 3 conv in each layer, the 3rd being dilated
            LayerModule = DenseLayer_Dil
        else:
            raise ValueError('Unknown type: %d' % layer_type)
        print('Dense channels:', num_input_features, end='')
        for i in range(num_layers):
            print(">", num_input_features + (i + 1) * growth_rate, end='')
            layer = LayerModule(
                num_input_features + i * growth_rate,
                kernels[i],
                params,
                first=i==0,
                )
            self.add_module('denselayer%d' % (i + 1), layer)
        
    def update(self, x):
        for layer in list(self.children()):
            x = layer.update(x)
        return x

    def reset_buffers(self):
        for layer in list(self.children()):
            layer.reset_buffers()

    def track(self, x):
        activations = []
        for layer in list(self.children()):
            # layer is a DenseLayer
            x, newf = layer.track(x)
            activations.append(newf.data.cpu().numpy())
            x = torch.cat([x, newf], 1)
        return x, activations


class DenseNet(nn.Module):
    def __init__(self, num_init_features, params):
        super(DenseNet, self).__init__()
        block_layers = params.get('num_layers', (24))
        block_kernels = params['kernels']
        growth_rate = params.get('growth_rate', 32)
        divide_channels = params.get('divide_channels', 2)
        init_weights = params.get('init_weights', 0)
        normalize_channels = params.get('normalize_channels', 0)
        transition_type = params.get('transition_type', 1)
        skip_last_trans = params.get('skip_last_trans', 0)

        if transition_type == 1:
            TransitionLayer = Transition
        elif transition_type == 2:
            TransitionLayer = Transition2

        self.features = nn.Sequential()
        num_features = num_init_features
        # start by normalizing the input channels #FIXME
        if normalize_channels:
            self.features.add_module('initial_norm',
                                     nn.GroupNorm(1, num_features))

        # start by reducing the input channels
        if divide_channels > 1:
            # In net2: trans = TransitionLayer
            trans = nn.Conv2d(num_features, num_features // divide_channels, 1)
            if init_weights == "manual":
                std = sqrt(1 / num_features)
                trans.weight.data.normal_(0, std)
            self.features.add_module('initial_transition', trans)
            num_features = num_features // divide_channels
        # Each denseblock
        for i, (num_layers, kernels) in enumerate(zip(block_layers,
                                                      block_kernels)):
            block = DenseBlock(num_layers, num_features,
                                kernels, params)
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            # In net2: Only between blocks
            if not i == len(block_layers) - 1 or not skip_last_trans:
                trans = TransitionLayer(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                    init_weights=init_weights)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                print("> (trans) ", num_features, end='')
        print()
        self.output_channels = num_features
        # Final batch norm
        self.features.add_module('norm_last', nn.BatchNorm2d(num_features))
        self.features.add_module('relu_last', nn.ReLU(inplace=True))

    def forward(self, x):
        return self.features(x.contiguous())

    def update(self, x):
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, DenseBlock):
                x = layer.update(x)
            else:
                x = layer(x)
        return x

    def reset_buffers(self):
        for layer in list(self.features.children()):
            if isinstance(layer, DenseBlock):
                layer.reset_buffers()

    def track(self, x):
        activations = []
        x = x.contiguous()
        for layer in list(self.features.children()):
            if isinstance(layer, DenseBlock):
                x, actv = layer.track(x)
                activations.append(actv)
            else:
                x = layer(x)
        return x, activations