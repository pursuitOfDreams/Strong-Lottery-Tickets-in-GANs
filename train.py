import time
import torch
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn as nn
import os
from tqdm import tqdm
from pruner import prune_model,remove_pruning
import matplotlib.pyplot as plt 

def extractFeatures(batchOfData, curNetEnc, detachOutput=False):
    # gets features from each layer of netEnc
    ftrs = []
    ftrsPerLayer = curNetEnc(batchOfData)[1]
    numFeaturesForEachEncLayer = curNetEnc.numberOfFeaturesPerLayer 
    numLayersToFtrMatching = min(16, len(numFeaturesForEachEncLayer))
    for lId in range(1, numLayersToFtrMatching + 1):
        cLid = lId - 1  # gets features in forward order
        
        ftrsOfLayer = ftrsPerLayer[cLid].view(
            ftrsPerLayer[cLid].size()[0], -1)
        
        if detachOutput:
            ftrs.append(ftrsOfLayer.detach())
        else:
            ftrs.append(ftrsOfLayer)
    ftrs = torch.cat(ftrs, dim=1)
    return ftrs

def compute_real_features(dataloader, curNetEnc, device, numExamplesProcessed):
    input_t = torch.FloatTensor(64, 3, 28, 28).to(device)
    if numExamplesProcessed is None:
        numExamplesProcessed = 0.0
    globalFtrMeanValues = []
    for i, data in enumerate(tqdm(dataloader), 1):
        real_cpu = data[0]      # img, target

        real_cpu = real_cpu.to(device)

        if real_cpu.shape[1] ==1:
            real_cpu = real_cpu.expand(
                real_cpu.shape[0], 3, real_cpu.shape[-1], real_cpu.shape[-1])

        input_t.resize_as_(real_cpu).copy_(real_cpu)
        realData = Variable(input_t)
        numExamplesProcessed += realData.size()[0]

        # extracts features for TRUE data
        allFtrsTrue = extractFeatures(realData, curNetEnc, detachOutput=True)

        if len(globalFtrMeanValues) < 1:
            globalFtrMeanValues = torch.sum(allFtrsTrue, dim=0).detach()
            featureSqrdValues = torch.sum(allFtrsTrue** 2, dim=0).detach()
        else:
            globalFtrMeanValues += torch.sum(allFtrsTrue, dim=0).detach()
            featureSqrdValues += torch.sum(allFtrsTrue ** 2, dim=0).detach()

    return numExamplesProcessed, globalFtrMeanValues, featureSqrdValues

def computeReal( dataloader, curNetEnc, device):
    numExamplesProcessed = 0.0
    numExamplesProcessed, globalFtrMeanValues, featureSqrdValues = compute_real_features(
                                                                    dataloader=dataloader,
                                                                    curNetEnc = curNetEnc,
                                                                    device=device,
                                                                    numExamplesProcessed=numExamplesProcessed)
    # variance = (SumSq - (Sum x Sum) / n) / (n - 1)
    globalFtrVarValues = (featureSqrdValues - (globalFtrMeanValues ** 2) / numExamplesProcessed) / (
                        numExamplesProcessed - 1)
    globalFtrMeanValues = globalFtrMeanValues / numExamplesProcessed
    return globalFtrMeanValues, globalFtrVarValues

def mmdtrain( netG, curNetEnc, netMean, netVar, optimizerG, optimizerMean, optimizerVar, traindataloader, epoch, device):
        
    netG.train()
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=10, eta_min=0)
    schedulerMean = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerMean, T_max=10, eta_min=0)
    schedulerVar = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerVar, T_max=10, eta_min=0)
  
    noise = torch.FloatTensor(64, 128).to(device)
    fixed_noise = Variable(torch.FloatTensor(1, 128).normal_(0, 1).to(device))
   
    avrgLossNetGMean = 0.0
    avrgLossNetGVar = 0.0
    avrgLossNetMean = 0.0
    avrgLossNetVar = 0.0

    criterionL2Loss = nn.MSELoss().to(device)
    FID_list = []

    features_dict = 'sngan_19_mnist.tar'

    if os.path.isfile("./features/"+features_dict) == True:
        ftr_check = torch.load("./features/"+features_dict, map_location = device)
        globalFtrMeanValues, globalFtrVarValues = ftr_check['Mean'].to(device), ftr_check['Var'].to(device)
    else:
        globalFtrMeanValues, globalFtrVarValues = computeReal(traindataloader, curNetEnc, device) 
        torch.save({'Mean':globalFtrMeanValues, 'Var':globalFtrVarValues}, "./features/"+features_dict)

    # Training start
    start_time = time.time()
  
    for iterId in tqdm(range(1000)):
        curNetEnc.zero_grad()
        netG.zero_grad()
        netMean.zero_grad()
        netVar.zero_grad()
        noise.resize_(64, 128).normal_(0, 1.0)
        noisev = Variable(noise)
      
        fakeData = netG(noisev)
        input_t = torch.FloatTensor(64, 3, 64, 64).to(device)
        # real_cpu = data[0]
        # input_t.resize_as_(real_cpu).copy_(real_cpu)
        # realData = Variable(input_t)


        fakeData = fakeData.expand(fakeData.shape[0], 3, fakeData.shape[-1], fakeData.shape[-1])

        ftrsFake = [extractFeatures( fakeData, curNetEnc, detachOutput=False)] #featureextract

        # updates Adam moving average of mean differences
            
        ftrsMeanFakeData = [torch.mean(ftrsFakeData, 0) for ftrsFakeData in ftrsFake]#evaluate mean
        diffFtrMeanTrueFake = globalFtrMeanValues.detach() - ftrsMeanFakeData[0].detach()

        lossNetMean = criterionL2Loss(netMean.weight, diffFtrMeanTrueFake.detach().view(1, -1))

        lossNetMean.backward()
        avrgLossNetMean += lossNetMean.item()
        optimizerMean.step()
        # if opt.scheduler != False:
        schedulerMean.step()


        # updates moving average of variance differences
        ftrsVarFakeData = [torch.var(ftrsFakeData, 0) for ftrsFakeData in ftrsFake]
        diffFtrVarTrueFake = globalFtrVarValues.detach() - ftrsVarFakeData[0].detach()

        lossNetVar = criterionL2Loss(netVar.weight, diffFtrVarTrueFake.detach().view(1, -1))

        lossNetVar.backward()
        avrgLossNetVar += lossNetVar.item()
        optimizerVar.step()
        schedulerVar.step()


        # updates generator
        meanDiffXTrueMean = netMean(globalFtrMeanValues.view(1, -1)).detach()
        meanDiffXFakeMean = netMean(ftrsMeanFakeData[0].view(1, -1))

        varDiffXTrueVar = netVar(globalFtrVarValues.view(1, -1)).detach()
        varDiffXFakeVar = netVar(ftrsVarFakeData[0].view(1, -1))



        lossNetGMean = (meanDiffXTrueMean - meanDiffXFakeMean)
        avrgLossNetGMean += lossNetGMean.item()

        lossNetGVar = (varDiffXTrueVar - varDiffXFakeVar)
        avrgLossNetGVar += lossNetGVar.item()

        regularization_loss = torch.tensor(0).to(device)
     
        lossNetG = lossNetGMean + lossNetGVar + regularization_loss
  

        lossNetG.backward()
        optimizerG.step()
        # if opt.scheduler != False:
        schedulerG.step()

            
        avrgLossNetGMean = 0.0
        avrgLossNetMean = 0.0
        avrgLossNetGVar = 0.0
        avrgLossNetVar = 0.0
            

            
        # saving models
        if (iterId + 1) % 100== 0:
            # if opt.algorithm in ["ep", "global_ep"]:
            fakeData = netG(fixed_noise).detach().cpu()
            plt.imsave(f'./generated_out/MNIST/sngan/{epoch}_{iterId}_fake_image.png' , fakeData[0,0])
            print("loss of netG: ",lossNetG.item())


    # prune_model(netG)
    print('#'*40)
    print('Finish Training')
    print("Running Time: ", time.time() - start_time)
    print('#'*40)
    # print("Best performance(FID, idx): ", min(FID_list), 100 * (FID_list.index(min(FID_list))+1))

def vggtrain(vgg, trainloader, testloader, epochs):
    prev = 0
    curr = 0
    for epoch in range(epochs):
        train_loss = 0
        train_acc = 0 
        valid_loss = 0
        valid_acc = 0
        criterion =  nn.NLLLoss()
        correct =0
        total_examples =0
        optimizer = torch.optim.Adam(vgg.parameters(), lr = 0.0001)

        vgg.train()
        start = time.time()
        prev = curr

        for idx, (data, target) in enumerate(tqdm(trainloader)):
            optimizer.zero_grad()
            pred, _ = vgg(data.repeat(1,3,1,1))
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, pred = torch.max(pred, dim = 1)
            correct += torch.sum(pred==target).item()
            total_examples += len(target)

        curr = 100*float(correct)/total_examples
        # evaluation
        with torch.no_grad():
            crrt = 0
            te =0
            for x,y in testloader:
                pred, _ = vgg(x.repeat(1,3,1,1))
                _, pred = torch.max(pred, dim = 1)
                crrt += torch.sum(pred==y).item()
                te += len(y)
        
        if curr > prev:
            torch.save(vgg, f"checkpoints/models/vgg/vgg_{epoch}.pth")
            print("saving model")
        print(f"Epoch: {epoch}\t train_accuracy: {100*float(correct)/total_examples}\t test_accuracy: {100*(float(crrt)/te)}")



