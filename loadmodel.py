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

CUR_CLASS = 0
THUMBNAIL_SIZE = (200,200)
IMAGE_SIZE = (800,800)
THUMBNAIL_PAD = (1,1)
ROOT_FOLDER = r'VOCdevkit/VOC2012/JPEGImages'
screen_size = [1280, 720]
thumbs_per_row = int(screen_size[0]/(THUMBNAIL_SIZE[0]+THUMBNAIL_PAD[0])) - 1
thumbs_rows = int(screen_size[1]/(THUMBNAIL_SIZE[1]+THUMBNAIL_PAD[1])) - 1
THUMBNAILS_PER_PAGE = (thumbs_per_row, thumbs_rows)


def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im
def convert_to_bytes(file_or_bytes, resize=None, fill=False):
    '''
    Will convert into bytes and optionally resize an image that is a file or a base64 bytes object.
    Turns into  PNG format in the process so that can be displayed by tkinter
    :param file_or_bytes: either a string filename or a bytes base64 image object
    :type file_or_bytes:  (Union[str, bytes])
    :param resize:  optional new size
    :type resize: (Tuple[int, int] or None)
    :return: (bytes) a byte-string object
    :rtype: (bytes)
    '''
    if isinstance(file_or_bytes, str):
        img = PIL.Image.open(file_or_bytes)
    else:
        try:
            img = PIL.Image.open(io.BytesIO(base64.b64decode(file_or_bytes)))
        except Exception as e:
            dataBytesIO = io.BytesIO(file_or_bytes)
            img = PIL.Image.open(dataBytesIO)

    cur_width, cur_height = img.size
    if resize:
        new_width, new_height = resize
        scale = min(new_height / cur_height, new_width / cur_width)
        img = img.resize((int(cur_width * scale), int(cur_height * scale)), PIL.Image.ANTIALIAS)
    if fill:
        img = make_square(img, THUMBNAIL_SIZE[0])
    with io.BytesIO() as bio:
        img.save(bio, format="PNG")
        del img
        return bio.getvalue()
def display_image_window(filename,f):
    try:
        index = f.findIndex(filename)
        str = ""
        for i in range(20):
            str += f"{classes[i]} - prediction: {round(float(f.pictures[index].prediction[i]),2)}, label: {f.pictures[index].label[i]}\n"

        layout = [[sg.Image(data=convert_to_bytes(filename, IMAGE_SIZE), enable_events=True)],
        [sg.Text(str)]]
        e,v = sg.Window(filename, layout, modal=True, element_padding=(0,0), margins=(0,0)).read(close=True)
    except Exception as e:
        print(f'** Display image error **', e)
        return
def make_thumbnails(flist):
    layout = [[sg.Text(f'Sorted on: {classes[CUR_CLASS]}', k='SORTED_ON', size = (20, 1))],[]]
    for row in range(THUMBNAILS_PER_PAGE[1]):
        row_layout = []
        for col in range(THUMBNAILS_PER_PAGE[0]):
            try:
                f = flist[row*THUMBNAILS_PER_PAGE[1] + col]
                row_layout.append(sg.B('',k=(row,col), size=(0,0), pad=THUMBNAIL_PAD,))
            except:
                pass
        layout += [row_layout]
    layout += [
    [sg.B(sg.SYMBOL_LEFT + ' Prev', size=(10,3), k='-PREV-'), sg.B('Next '+sg.SYMBOL_RIGHT, size=(10,3), k='-NEXT-'), sg.B('Exit', size=(10,3)), sg.Slider((0,100), orientation='h', size=(50,15), enable_events=True, key='-SLIDER-')],
    [sg.B(sg.SYMBOL_LEFT + ' Prev Class', size=(10,3), k='-PREVCLASS-'),
     sg.B('Next Class '+sg.SYMBOL_RIGHT, size=(10,3), k='-NEXTCASS-')]
    ]
    return sg.Window('Thumbnails', layout, element_padding=(0, 0), margins=(0, 0), finalize=True, grab_anywhere=False, location=(0,0), return_keyboard_events=True)
def display_images(t_win, offset, files):
    currently_displaying = {}
    row = col = 0
    while True:
        if offset + 1 > len(files) or row == THUMBNAILS_PER_PAGE[1]:
            break
        f = files[offset]
        currently_displaying[(row, col)] = f
        try:
            t_win[(row, col)].update(image_data=convert_to_bytes(f, THUMBNAIL_SIZE, True))
        except Exception as e:
            print(f'Error on file: {f}', e)
        col = (col + 1) % THUMBNAILS_PER_PAGE[0]
        if col == 0:
            row += 1

        offset += 1
    if not (row == 0 and col == 0):
        while row != THUMBNAILS_PER_PAGE[1]:
            t_win[(row, col)].update(image_data=sg.DEFAULT_BASE64_ICON)
            currently_displaying[(row, col)] = None
            col = (col + 1) % THUMBNAILS_PER_PAGE[0]
            if col == 0:
                row += 1


    return offset, currently_displaying


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
    def findIndex(self, filename):
        for i in range(len(self.pictures)):
            if self.pictures[i].filename == filename:
                return i


p = []
for i in range(len(fnames)):
    p.append(Picture(os.path.join(ROOT_FOLDER, fnames[i]), concat_labels[i], concat_pred[i]))

f = Files(p)
f.sortOnClass(0,True)
files = f.fnamelist()
t_win = make_thumbnails(files)
offset, currently_displaying = display_images(t_win, 0, files)
offset = THUMBNAILS_PER_PAGE[0] * THUMBNAILS_PER_PAGE[1]

while True:
    win, event, values = sg.read_all_windows()
    if win == sg.WIN_CLOSED:            # if all windows are closed
        break

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

    if isinstance(event, tuple):
        display_image_window(currently_displaying.get(event),f)
        continue
    elif event == '-SLIDER-':
        offset = int(values['-SLIDER-']*len(files)/100)
        event = '-NEXT-'
    else:
        t_win['-SLIDER-'].update(offset * 100 / len(files))

    if event == '-NEXT-' or event.endswith('Down'):
        offset, currently_displaying = display_images(t_win, offset, files)
    elif event == '-PREV-' or event.endswith('Up'):
        offset -= THUMBNAILS_PER_PAGE[0]*THUMBNAILS_PER_PAGE[1]*2
        if offset < 0:
            offset = 0
        offset, currently_displaying = display_images(t_win, offset, files)
    if event == '-PREVCLASS-':
        CUR_CLASS -= 1
        if CUR_CLASS <= 0:
            CUR_CLASS = 19
        f.sortOnClass(CUR_CLASS, True)
        files = f.fnamelist()
        t_win.Element('SORTED_ON').Update(f'Sorted on: {classes[CUR_CLASS]}')
        offset, currently_displaying = display_images(t_win, 0, files)

    if event == '-NEXTCASS-':
        CUR_CLASS += 1
        if CUR_CLASS >= 20:
            CUR_CLASS = 0
        f.sortOnClass(CUR_CLASS, True)
        files = f.fnamelist()
        t_win.Element('SORTED_ON').Update(f'Sorted on: {classes[CUR_CLASS]}')
        offset, currently_displaying = display_images(t_win, 0, files)
