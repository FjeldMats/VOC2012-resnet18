import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import functional as F

import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # to work on x11 forwarding

from torch import Tensor

import time
import timeit
from datetime import datetime
import os
import numpy as np

import PIL.Image
import sklearn.metrics

from vocparseclslabels import PascalVOC

from typing import Callable, Optional


class dataset_voc(Dataset):
  def __init__(self, root_dir, trvaltest, transform=None):

     # what dataset to get
    if(trvaltest == 0):
        trvaltest = 'train'
    if(trvaltest == 1):
        trvaltest = 'val'

    self.transform = transform
    # create txt files and combine every image into dataframe with all classes
    self.pv=PascalVOC(root_dir)
    self.df = self.pv._imgs_from_category('aeroplane', trvaltest).set_index('filename').rename({'true': 'aeroplane'}, axis=1)
    for c in self.pv.list_image_sets()[1:]:
        ls = self.pv._imgs_from_category(c, trvaltest).set_index('filename')
        ls = ls.rename({'true': c}, axis=1)
        self.df = self.df.join(ls, on="filename")

    # filenames are index into the dataframe
    self.imgfilenames = self.df.index.values

  def __len__(self):
      return len(self.imgfilenames)

  def __getitem__(self, idx):

    image = PIL.Image.open(f"VOCdevkit/VOC2012/JPEGImages/{self.imgfilenames[idx]}.jpg").convert('RGB')

    classes = [1 if i == 0 else 0 if i==-1 else i for i in list(self.df.iloc[idx])]

    #apply transfroms to image
    if self.transform:
            image = self.transform(image)

    sample = {'image': image, 'label': classes, 'filename': self.imgfilenames[idx]}
    return sample



def train_epoch(model, trainloader, criterion, device, optimizer):

    model.train()

    losses = []
    for batch_idx, data in enumerate(trainloader):

        if (batch_idx%100==0) and (batch_idx>=100):
            print('at train batchindex: ', batch_idx)

        inputs = data['image'].to(device)

        #convert list of tensors to tensors
        labels = data['label']
        labels = torch.transpose(torch.stack(labels), 0, 1)
        labels = labels.type_as(inputs)

        labels = labels.to(device)

        optimizer.zero_grad()
        output = model(inputs)

        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.mean(losses)



def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    #TODO
    model.eval()

    curcount = 0
    accuracy = 0

    concat_pred=[np.empty(shape=(0)) for _ in range(numcl)] #prediction scores for each class. each numpy array is a list of scores. one score per image
    concat_labels=[np.empty(shape=(0)) for _ in range(numcl)] #labels scores for each class. each numpy array is a list of labels. one label per image
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    counter = [0]*numcl
    Ys = [[] for j in range(numcl)]
    ys = [[] for j in range(numcl)]
    with torch.no_grad():
      losses = []
      for batch_idx, data in enumerate(dataloader):

          if (batch_idx%100==0) and (batch_idx>=100):
               print('at val batchindex: ',batch_idx)

          inputs = data['image'].to(device)

          #convert list of tensors to tensors
          labels = data['label']
          labels = torch.transpose(torch.stack(labels), 0, 1)
          labels = labels.type_as(inputs)

          labels = labels.to(device)

          output = model(inputs)

          loss = criterion(output, labels.to(device) )
          losses.append(loss.item())

          m = nn.Sigmoid()
          threshold_output = (m(output)>0.5).float()
          cpuout = output.cpu()
          labels = labels.cpu()

          for c in range(numcl):
              Y = labels[:, c]
              y = m(cpuout[:, c])
              for i in range(len(Y)):
                  Ys[c].append(Y[i])
                  ys[c].append(y[i])

    for c in range(numcl):
        avgprecs[c] = sklearn.metrics.average_precision_score(Ys[c], ys[c])

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames

def traineval2_model_nocv(dataloader_train, dataloader_test, model, criterion, optimizer, scheduler, num_epochs, device, numcl):

  best_measure = 0
  best_epoch =-1

  trainlosses=[]
  testlosses=[]
  testperfs=[]
  bestweights=[]
  for epoch in range(num_epochs):
    start = timeit.default_timer()
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)

    avgloss = train_epoch(model, dataloader_train, criterion, device, optimizer)
    trainlosses.append(avgloss)

    if scheduler is not None:
      scheduler.step()

    perfmeasure, testloss,concat_labels, concat_pred, fnames = evaluate_meanavgprecision(model, dataloader_test, criterion, device, numcl)
    testlosses.append(testloss)


    print('at epoch: ', epoch,' classwise perfmeasure ', perfmeasure)
    print(f'train loss: {avgloss}, test loss {testloss}')

    avgperfmeasure = np.mean(perfmeasure)
    testperfs.append(avgperfmeasure)
    print('avgperfmeasure ', avgperfmeasure)
    stop = timeit.default_timer()
    #print(f'Epoch Finished in {stop - start}\n')

    if avgperfmeasure > best_measure and avgperfmeasure > 0.8:
      bestweights = model.state_dict()
      #TODO track current best performance measure and epoch
      torch.save(model.state_dict(), f"models/model_{datetime.now().strftime('%d-%m-%Y_%H:%M:%S')}_{round(float(avgperfmeasure),4)}.pth")
      best_measure = avgperfmeasure

      #TODO save your scores
  return best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs


class yourloss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(yourloss, self).__init__(None, None, reduction)
        self.register_buffer('weight', None)
        self.register_buffer('pos_weight', None)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, self.weight, pos_weight=None, reduction=self.reduction)

def runstuff():

  config = dict()

  config['use_gpu'] = True  #TODO change this to True for training on the cluster, eh
  config['lr'] = 1e-3
  config['batchsize_train'] = 32
  config['batchsize_val'] = 128
  config['maxnumepochs'] = 16
  #config['maxnumepochs'] = 10

  config['scheduler_stepsize'] = 10
  config['scheduler_factor'] = 0.2

  # kind of a dataset property
  config['numcl'] = 20

  #data augmentations
  data_transforms = {
      'train': transforms.Compose([
          transforms.Resize(256),
          transforms.RandomCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
      'val': transforms.Compose([
          transforms.Resize(224),
          transforms.CenterCrop(224),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
      ]),
  }

  #datasets
  image_datasets={}
  image_datasets['train']=dataset_voc(root_dir='VOCdevkit/VOC2012',trvaltest=0, transform=data_transforms['train'])
  image_datasets['val']=dataset_voc(root_dir='VOCdevkit/VOC2012',trvaltest=1, transform=data_transforms['val'])

  #dataloaders
  #TODO use num_workers=1
  bach_size = {"train":config['batchsize_train'],"val":config['batchsize_val']}
  dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], bach_size[x], shuffle=True) for x in ['train', 'val']}

  #device
  if True == config['use_gpu']:
      device= torch.device('cuda:0')
  else:
      device= torch.device('cpu')

  model = models.resnet18(pretrained=True) #pretrained resnet18
  for param in model.parameters():
      param.requires_grad = False

  #overwrite last linear layer
  num_ftrs = model.fc.in_features
  model.fc = nn.Linear(num_ftrs, config['numcl'])

  model = model.to(device)

  lossfct = yourloss()

  # Observe that all parameters are being optimized
  someoptimizer = optim.Adam(model.fc.parameters(), lr=config['lr'])

  # Decay LR by a factor of 0.3 every X epochs
  somelr_scheduler = lr_scheduler.StepLR(someoptimizer, step_size=config['scheduler_stepsize'], gamma=config['scheduler_factor'])

  best_epoch, best_measure, bestweights, trainlosses, testlosses, testperfs = traineval2_model_nocv(dataloaders['train'], dataloaders['val'] ,  model ,  lossfct, someoptimizer, somelr_scheduler, num_epochs = config['maxnumepochs'], device = device , numcl = config['numcl'])


  # plot loss
  plt.figure(0)
  plt.plot(range(config['maxnumepochs']), trainlosses, label="train")
  plt.plot(range(config['maxnumepochs']), testlosses, label="test")
  plt.title('Loss curve')
  plt.xlabel('epochs')
  plt.ylabel('loss')
  plt.legend(loc="upper right")
  plt.savefig('plots/traintest_loss.png')

  # plot pref
  plt.figure(1)
  plt.plot(range(config['maxnumepochs']), testperfs)
  plt.title('test prefs')
  plt.xlabel('epochs')
  plt.ylabel('mAP')
  plt.savefig('plots/mAP.png')


if __name__=='__main__':

  runstuff()
