from model.vgg import init_vgg
from train import vggtrain, mmdtrain
from dataset import trainloader, testloader, device
from model.generator import SNResGenerator
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchvision.utils import make_grid

vgg = torch.load("./checkpoints/models/vgg/vgg_19.pth")
netG = SNResGenerator(64).to(device)

epochs = 20

def createMovingNet(curNetEnc, device):
    numFeaturesInEnc = 0
    numFeaturesForEachEncLayer = curNetEnc.numberOfFeaturesPerLayer
    numLayersToFtrMatching = min(16, len(numFeaturesForEachEncLayer))
    numFeaturesInEnc += sum(numFeaturesForEachEncLayer[-numLayersToFtrMatching:])
    netMean = nn.Linear(numFeaturesInEnc, 1, bias=False)
    netVar = nn.Linear(numFeaturesInEnc, 1, bias=False)
    netMean.to(device)
    netVar.to(device)
    return netMean, netVar

def sample_images(fake_images,epoch):
    img_grid = make_grid(fake_images.detach().cpu(), nrows = 8)
    print(img_grid)
    plt.imsave(f'./generated_out/MNIST/sngan/{epoch}_fake_Data.png',((img_grid+1)/2).permute(1, 2, 0).numpy(), cmap='gray')

if __name__=="__main__":
    

    netMean, netVar = createMovingNet( vgg, device)

    optimizerG = optim.Adam(netG.parameters(), lr=5e-3  , betas=(0.5, 0.999))
    optimizerMean = optim.Adam(netMean.parameters(), lr=1e-4, betas=(0.5, 0.999))
    optimizerVar = optim.Adam(netVar.parameters(), lr=1e-4, betas=(0.5, 0.999))


    for epoch in range(50):
        print("here")
        mmdtrain(netG, vgg, netMean, netVar, optimizerG, optimizerMean, optimizerVar, trainloader , epoch, device = device)
        noise = torch.FloatTensor(1,128).to(device)
        noise.resize_(1,128).normal_(0, 1.0)

        noisev = Variable(noise)
        fakeData = netG(noisev).detach().cpu()
        plt.imsave(f'./generated_out/MNIST/sngan/{epoch}_fake_image.png' , fakeData[0,0])
        # sample_images(fakeData,epoch)

        
        torch.save(netG.state_dict() , f'./models/sngan/no/sngan_{epoch}.pth')