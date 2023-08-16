
import torch
import torch.nn as nn
import torch.nn.functional as F
from pruner import get_subnet

class subConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(subConv, self).__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad=True)
        if self.bias != None:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()), requires_grad = True)
        else:
            self.bias_scores = nn.Parameter(torch.tensor(0.9))
        nn.init.uniform_(self.scores, a=-0.1, b=0.1)
        nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

    def forward(self, x):
        weight_subnet = get_subnet(self.scores.abs(), 0.9)
        bias_subnet = get_subnet(self.bias_scores.abs(), 0.9)
        
        w = self.weight*weight_subnet
        b = self.bias*bias_subnet

        out = F.conv2d(
            x, w, b, self.stride, self.padding, self.dilation, self.groups
        )

        return out

class subLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(subLinear, self).__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()), requires_grad =True)

        if self.bias != None:
            self.bias_scores = nn.Parameter(torch.Tensor(self.bias.size()))
        else:
            self.bias_scores = nn.Parameter(torch.tensor(0.9))    
        nn.init.uniform_(self.scores, a=-0.1, b=0.1)
        nn.init.uniform_(self.bias_scores, a=-1.0, b=1.0)

    def forward(self, x):
        weight_subnet = get_subnet(self.scores.abs(), 0.9)
        bias_subnet = get_subnet(self.bias_scores.abs(), 0.9)
        
        w = self.weight*weight_subnet
        b = self.bias*bias_subnet

        return F.linear(x, w, b)
        
        
class GResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False):
        super(GResBlock, self).__init__()
        hidden_channels = in_channels
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()
    def forward_residual_connect(self, input):
        out = self.conv_sc(input)
        if self.upsample:
             out = self.upsampling(out)
            #out = self.upconv2(out)
        return out
    def forward(self, input):
        out = self.relu(self.bn1(input))
        out = self.conv1(out)
        if self.upsample:
             out = self.upsampling(out)
             #out = self.upconv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out_res = self.forward_residual_connect(input)
        return out + out_res

class SNResGenerator(nn.Module):
    def __init__(self, ngf, z=128, nlayers=4):
        super(SNResGenerator, self).__init__()
        self.input_layer = nn.Linear(z, (4 ** 2) * ngf * 16)
        self.generator = self.make_model(ngf, nlayers)

    def make_model(self, ngf, nlayers):
        model = []
        tngf = ngf*16
        for i in range(nlayers):
            model += [GResBlock(tngf, tngf//2, upsample=True)]
            tngf //= 2

        model += [nn.BatchNorm2d(ngf)]
        model += [nn.ReLU()]
        model += [nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)]
        model += [nn.Tanh()]
        return nn.Sequential(*model)

    def forward(self, z):
        out = self.input_layer(z)
        out = out.view(z.size(0), -1, 4, 4)
        out = self.generator(out)

        return out