import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from dataclasses import dataclass
from typing import Optional
import sys
import os
from datetime import datetime
from dataclasses import dataclass
import argparse

if '/content/drive/MyDrive/MultiModal' not in sys.path:# try with and without '/' at the end
  sys.path.append('/content/drive/MyDrive/MultiModal') # try with and without '/' at the end


import MultiModal
import PaligemmaProcessing

def saveModel(model, pathReq = None):

  fileName = 'model_' + str(datetime.now()).replace('-','').replace(':','').replace(' ','')[:14] + '.pth'

  if pathReq is None:
      filePath = os.path.join(os.getcwd(), fileName)
  else:
      filePath =  os.path.join(pathReq, fileName)

  torch.save(model, filePath)

  print('model saved in {}'.format(filePath))


@dataclass
class Configs:
  patchSize:int = 16
  embeddingChannels:int = 32
  batchSize: int = 2
  layerNormEps: int = 1e-5
  intermediateEmbedding :int = 32 * 4
  numLayers: int = 6
  numHeads: int = 4
  dropoutRate:float = 0.5
  training: bool = True
  imageDim:int = 28
  device:str = 'cuda'

@dataclass
class TextConfigs:
  vocabSize:int = 0
  visionEmbeddingChannels:int = 0
  textEmbeddingChannels:int = 0
  batchSize: int = 2
  rmsNormEps: int = 1e-5
  intermediateEmbedding :int = 32 * 4
  numLayers: int = 6
  numHeads: int = 4
  dropoutRate:float = 0.5
  training:bool = True
  base: int = 1000
  batchSize:int  = 8
  maxSeqLength:int = 32
  dropoutRate:float = 0.5
  device:str = 'cuda'


def runTraining():

  parser = argparse.ArgumentParser()
  parser.add_argument("--batchSize", required = False, type=int, default = 1)
  parser.add_argument("--learningRate", required = False, type=float, default = 1e-5)
  parser.add_argument("--epochs", required = False, type=int, default = 5)
  parser.add_argument("--parallelHeads", required = False, type=int, default = 4) 
  parser.add_argument("--nImageLayers", required = False, type=int, default = 1) #how many blocks to use to make deeper network
  parser.add_argument("--patchSize", required = False, type=int, default = 1) 
  parser.add_argument("--nTextLayers", required = False, type=int, default = 2)    				
  parser.add_argument("--imageChannels", required = False, type=int, default = 4) 
  parser.add_argument("--textChannels", required = False, type=int, default = 512) 
  parser.add_argument("--usedProcessor", required = False, default = 'cuda')
  parser.add_argument("--savePath", required = False, default = None)

  args = parser.parse_args()
  print('\n\n' + '=' * 40 + ' Starting training with parameters: ', args, '=' * 40 + '\n\n' )


  transform = transforms.Compose([	  
    transforms.Grayscale(num_output_channels=3),  # Ensure 3 channels (if needed)
    transforms.ToTensor(),  # Convert to tensor
  ])

  to_pil = transforms.ToPILImage()

  gBatchSize = args.batchSize 
  gDevice = args.usedProcessor
  train_data = torchvision.datasets.MNIST("./", train=True, transform=transform, download=True)
  test_data_xy = torchvision.datasets.MNIST("./", train=False, transform=transform, download=True)

  batch_size = gBatchSize
  trainLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
  testLoader = torch.utils.data.DataLoader(test_data_xy, batch_size=batch_size, shuffle=False, num_workers=2)
  len(trainLoader)

  imageDim = 28
  
  allChars =[l for l in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz1234567890!@#$%^&*()?,./;\n']
  vocab = sorted(list(set(allChars)))
  tokenMap = {k:v for v, k in enumerate(vocab)}
  charsAvailable = len(tokenMap.keys())
  tokenMap['<image>'] =  charsAvailable
  tokenMap['<bos>'] = charsAvailable + 1
  processing = PaligemmaProcessing.CharacterProcessor(tokenMap, imageDim * imageDim, imageDim)
  wordToList = lambda sInput, tokenMapping: [tokenMapping[letter] for letter in sInput] # to get numerical tensor to feed into nn.Embedding function (each letter has it's index)
  tensorToWord = lambda sIndexes, tokenMapping: [list(tokenMapping.keys())[i.item()] for i in sIndexes]

  iConf = MultiModal.Configs #vision configs
  iConf.embeddingChannels = args.imageChannels
  iConf.batchSize = gBatchSize
  iConf.patchSize = args.patchSize
  iConf.numLayers = args.nImageLayers
  iConf.numHeads = args.parallelHeads
  iConf.device = gDevice


  tConf = MultiModal.TextConfigs
  tConf.batchSize = gBatchSize
  tConf.vocabSize = len(tokenMap.keys())
  tConf.visionEmbeddingChannels = iConf.embeddingChannels #this is so we just need to pass one config to the multiModalProjector
  tConf.textEmbeddingChannels = args.textChannels
  tConf.numLayers = args.nTextLayers
  tConf.numHeads = args.parallelHeads
  tConf.device = gDevice

  model1 = MultiModal.MultiModalPipeline(tConf, iConf, tokenMap).to(device=gDevice)
  promptStrings = ['Which digit is this?'] * gBatchSize
  batchNumber = 0

  optimizer = torch.optim.Adam(model1.parameters(), lr=args.learningRate) #make optimizer for the decoder block
  torch.autograd.set_detect_anomaly(True)
  epochs = 10
  
  print('\n Using processor: ', gDevice)

	# Assuming your dataset is defined as trainDataset
	# Define your transformations

  for epoch in range(epochs):
    batchNumber = 0  # To count the batch number
      
    for x, y in trainLoader:
      # Preprocessing step
      pil_images = [torchvision.transforms.functional.to_pil_image(x[i,...]) for i in range(x.shape[0])]
      labelStrings = ['The digit is ' + str(yy.item()) for yy in y]
      inputs = [wordToList(p, tokenMap) for p in promptStrings]
      labels = torch.tensor([wordToList(l, tokenMap) for l in labelStrings],dtype=torch.long)
      inputDict = processing(inputs, pil_images)
      t = 0
      accum = 0
      while t < labels.shape[1]:
        if t > 0:
          inputIds = torch.cat((inputIds.to(device=gDevice), outLogits.to(device=gDevice)), dim = 1)
          attMask = torch.cat((attMask.to(device=gDevice), torch.ones(outLogits.shape, dtype=torch.long).to(device=gDevice)), dim = 1)
        else:
          inputIds = inputDict['inputs'].to(device=gDevice)
          attMask = torch.ones(inputIds.shape).to(device=gDevice)

        #generate 1 token
        predictions, loss = model1(
            inputIds,
            inputDict['pixel_values'].to(device=gDevice),
            attMask,
            labels=labels.clone()[:,t].unsqueeze(1)
        )
        outLogits = torch.argmax(predictions.clone().detach(), dim=-1).unsqueeze(1).clone().detach()
        if t == 0:
          outputString = outLogits
        else:
          outputString = torch.cat((outputString, outLogits), dim = 1)

        t += 1
        accum += loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

      # Track total loss for the epoch
      batchNumber += 1

      # Print out the loss every few batches (optional)
      if batchNumber % 5 == 0:
          print(f"Batch {batchNumber}: Accumulated loss = {accum.item()}: Loss = {loss.item()}: Sample labels: {y}: Test Output:  {[''.join(tensorToWord(o, tokenMap)) for o in outputString]}")

    # Print average loss after each epoch
    print('====', torch.mean(torch.tensor(accum)), '=====', 'Sample label: ', y, 'Test Output: ', [''.join(tensorToWord(o, tokenMap)) for o in outputString])
    saveModel(model, pathReq = args.savePath)

if __name__ == '__main__':
	runTraining()


