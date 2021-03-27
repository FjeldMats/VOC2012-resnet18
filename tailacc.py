import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.nn import functional as F

import torchvision
from torchvision import datasets, models, transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib
matplotlib.use('TkAgg') # to work on x11 forwarding

from torch import Tensor

import time
import os
import numpy as np

import PIL.Image
import sklearn.metrics

from vocparseclslabels import PascalVOC
import PySimpleGUI as sg
import PIL
from PIL import Image
import io
import base64
import os


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

    classes = [1 if i == 0 else 0 if i==-1 else i for i in list(self.df.iloc[idx])]  # column contain all

    if self.transform:
            image = self.transform(image)

    sample = {'image': image, 'label': classes, 'filename': self.imgfilenames[idx]}
    return sample

class yourloss(nn.modules.loss._Loss):

    def __init__(self, reduction: str = 'mean') -> None:
        super(yourloss, self).__init__(None, None, reduction)
        self.register_buffer('weight', None)
        self.register_buffer('pos_weight', None)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, self.weight, pos_weight=None, reduction=self.reduction)


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

config = dict()
config['batchsize_train'] = 16
config['batchsize_val'] = 64
config['use_gpu'] = True
config['numcl'] = 20

#datasets
image_datasets={}
image_datasets['val']=dataset_voc(root_dir='VOCdevkit/VOC2012',trvaltest=1, transform=data_transforms['val'])

print("dataset")

#dataloaders
dataloader = torch.utils.data.DataLoader(image_datasets['val'], config['batchsize_val'], shuffle=True)

print("dataloader")

if True == config['use_gpu']:
    device= torch.device('cuda:0')

else:
    device= torch.device('cpu')

model = models.resnet18(pretrained=True) #pretrained resnet18
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, config['numcl'])

model.load_state_dict(torch.load("models/model0.8151495741589873.pth"))
model = model.to(device)
print("loaded model")

def evaluate_meanavgprecision(model, dataloader, criterion, device, numcl):

    model.eval()

    concat_pred = []
    concat_labels = []
    avgprecs=np.zeros(numcl) #average precision for each class
    fnames = [] #filenames as they come out of the dataloader

    counter = [0]*numcl
    Ys = [[] for j in range(numcl)]
    ys = [[] for j in range(numcl)]
    with torch.no_grad():
      losses = []
      print('eval',end='')
      for batch_idx, data in enumerate(dataloader):
          if batch_idx % 10 == 0:
              print('.',end='')
          #convert list of tensors to tensors
          labels = data['label']
          inputs = data['image'].to(device)
          labels = torch.transpose(torch.stack(labels), 0, 1)
          labels = labels.type_as(inputs)

          labels = labels.to(device)

          output = model(inputs)

          loss = criterion(output, labels.to(device) )
          losses.append(loss.item())

          m = nn.Sigmoid()
          #threshold_output = (m(output)>0.5).float()
          cpuout = output.cpu()
          labels = labels.cpu()

          for c in range(numcl):
              Y = labels[:, c]
              y = m(cpuout[:, c])
              for i in range(len(Y)):
                  Ys[c].append(Y[i])
                  ys[c].append(y[i])

          # save some data
          for fname, label, pred in zip(data['filename'], labels, cpuout):
              fnames.append(f"{fname}.jpg")
              concat_labels.append(label)
              concat_pred.append(m(pred))

    for c in range(numcl):
        avgprecs[c] = sklearn.metrics.average_precision_score(Ys[c], ys[c])

    return avgprecs, np.mean(losses), concat_labels, concat_pred, fnames

classes = ['aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train',
    'tvmonitor']

avgprecs, loss, concat_labels, concat_pred, fnames = evaluate_meanavgprecision(model, dataloader, yourloss(), device, config['numcl'])

class Picture():
    def __init__(self, filename, label, prediction):
        self.filename = filename
        self.label = label
        self.prediction = prediction

class Files():
    def __init__(self, pictures):
        self.pictures = pictures
        self.classpictures = []
    def __len__(self):
        return len(self.pictures)
    def sortOnClass(self, class_idx, rev):
        self.pictures.sort(key=lambda x: x.prediction[class_idx], reverse = rev)
    def fnamelist(self):
        lst = []
        for p in self.pictures:
            lst.append(p.filename)
        return lst

p = []
ROOT_FOLDER = r'VOCdevkit/VOC2012/JPEGImages'
for i in range(len(fnames)):
    p.append(Picture(os.path.join(ROOT_FOLDER, fnames[i]), concat_labels[i], concat_pred[i]))

def tailacc(t, preds, labels, c):
    num_above_t = 0
    num_above_t_c = 0
    for i in range(len(preds)):
        if preds[i][c] > t:
            num_above_t += 1
            if labels[i][c] == 1:
                num_above_t_c += 1
    return (1/num_above_t) * num_above_t_c



for c in range(20):
    max_pred_class = concat_pred[0][c]
    for pred in concat_pred:
        if pred[c] > max_pred_class:
            max_pred_class = pred[c]
    step = (max_pred_class - 0.5)/20
    t = 0.5
    ts = []
    tailaccs = []
    while t < max_pred_class:
        ts.append(float(t))
        tailaccs.append(tailacc(t, concat_pred, concat_labels, c))
        t += step

    str = f"{classes[c]}"


    plt.figure(c)
    plt.xlabel('t')
    plt.ylabel('tailacc(t)')
    #plt.legend(loc="lower right")
    plt.title(f"{classes[c]}")
    plt.plot(ts, tailaccs, label = str)
    plt.savefig(f"plots/tail_{classes[c]}.png")
