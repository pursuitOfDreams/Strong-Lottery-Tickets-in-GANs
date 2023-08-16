from collections import OrderedDict
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torchvision.models import vgg19_bn
import os 

os.environ['TORCH_HOME'] = "./models"
device ="cuda" if torch.cuda.is_available() else "cpu"

class VGG(nn.Module):
  def __init__(self, num_classes =10):
    super().__init__()
    self.num_classes = num_classes 
    self.vgg19 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19_bn', pretrained=True)
    for param in self.vgg19.parameters():
      param.requires_grad = False
    self.classifier = nn.Sequential(
       nn.Linear(1000, 256),
       nn.ReLU(),
       nn.Dropout(0.4),
       nn.Linear(256, 10),
       nn.LogSoftmax(dim = 1)
    )
    self.linear = nn.Linear(1000, num_classes)
    self.Out = OrderedDict()

  def forward(self,x ):
    out = self.vgg19(x)
    out = self.classifier(out)
    Out = []
    for k, v in self.Out.items():
        Out.append(v)
    Out.append(out)
    return out, Out
  
  def _get_hook(self, layer_num):
    Out = self.Out
    def myhook(module, _input, _out):
        Out[layer_num] = _out
    self.vgg19.features[layer_num].register_forward_hook(myhook)

def getNetImageSizeAndNumFeats(net, verbose=False, image_size = 32, use_cuda = True, nc = 3):
    if use_cuda:
        y, layers = net(Variable(torch.randn(1,nc,image_size,image_size).cuda()))
    else:
        y, layers = net(Variable(torch.randn(1,nc,image_size,image_size)))


    layer_img_size = []
    layer_num_feats = []
    for L in reversed(layers):
        if len(L.size()) == 4:
            layer_img_size.append(L.size(2))
            layer_num_feats.append(L.size(1)*L.size(2)*L.size(3))
        elif len(L.size()) == 3:    # vit
            print('L size', L.shape)
            layer_img_size.append(1)
            layer_num_feats.append(L.size(1)*L.size(2))

        elif len(L.size()) == 2:
            layer_img_size.append(1)
            layer_num_feats.append(L.size(1))
        elif len(L.size()) == 1:
            layer_num_feats.append(L.size(0))
        else:
            assert 0, 'not sure how to handle this layer size '+L.size()

    return layer_img_size, layer_num_feats

def init_vgg(num_classes =10):
   vgg = VGG(num_classes).to(device)
   for idx in range(len(vgg.vgg19.features)):
    if str(vgg.vgg19.features[idx])[0:4] == 'ReLU':
        vgg._get_hook(idx)
    ImgSizeL, numFeatsL = getNetImageSizeAndNumFeats(vgg, image_size=32)
    vgg.ImageSizePerLayer = np.array(ImgSizeL)
    vgg.numberOfFeaturesPerLayer = np.array(numFeatsL)
    return vgg