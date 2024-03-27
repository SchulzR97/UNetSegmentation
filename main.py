from data.dataloader import MSCocoLoader
from data.dataset import MSCocoDataset
import torch
from model import SegNet
from model import UNet
import gc
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import loss as L
from torch.utils.tensorboard import SummaryWriter
import copy
import cv2 as cv

device = 'mps'
batch_size = 4#128
epochs = 10000
avg_episodes = 100

learning_rate: float = 1e-6#1e-5
weight_decay: float = 1e-8
momentum: float = 0.999

num_categories = 20#90
size = (320, 320)
ds_train = MSCocoDataset(phase = 'train', size=None, device=device, input_size=size, num_categories=num_categories,
                         max_rot=20, min_scale=1.1, max_scale=1.5)
ds_val = MSCocoDataset(phase = 'val', size=None, device=device, input_size=size, num_categories=num_categories)
model = UNet(device, num_categories=num_categories, scale_size=4)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=False)
#criterion = torch.nn.BCEWithLogitsLoss()
#criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
#criterion = torch.nn.BCEWithLogitsLoss()
#criterion = L.DiceLoss(reduction='mean')

train_loader = torch.utils.data.DataLoader(
    ds_train,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
)
val_loader = torch.utils.data.DataLoader(
    ds_val,
    batch_size=batch_size,
    num_workers=0,
    pin_memory=True,
    shuffle=True,
)

writer = SummaryWriter()

epoch = 0

def train(model:SegNet, data_loader:torch.utils.data.DataLoader, optimizer, criterion, batch_size, iterations):
    model.train()

    loss_cum = 0
    iterator = iter(data_loader)
    for iteration in tqdm(range(iterations), desc='train'):
        X, T = next(iterator)

        optimizer.zero_grad()
        Y = model(X)
        #Y_mask = Y[]
        #dice_loss = loss.dice_loss(Y, T)
        loss = criterion(Y, T)
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(Y, T)
        #loss = torch.nn.functional.cross_entropy(Y, T)
        loss.backward()
        optimizer.step()
        loss_cum += loss.item()
    return loss_cum / iterations

def val(model:SegNet, data_loader:torch.utils.data.DataLoader, criterion, batch_size, iterations):
    model.eval()

    loss_cum = 0
    iterator = iter(data_loader)
    for iteration in tqdm(range(iterations), desc='val'):
        X, T = next(iterator)

        with torch.no_grad():
            Y = model(X)
        loss = criterion(Y, T)
        #loss = torch.nn.functional.binary_cross_entropy_with_logits(Y, T)
        #loss = torch.nn.functional.cross_entropy(Y, T)
        loss_cum += loss.item()
    return loss_cum / iterations

def vis_output(cat:int, writer, X, Y, epoch):
    min = Y[0, cat-1].min()
    max = Y[0, cat-1].max()

    img_Y = Y[0][cat-1]
    #img_Y = img_Y.unsqueeze(0)#((1, img_Y.shape[0], img_Y.shape[1]))
    mask = (img_Y > 0.5)#.squeeze()

    mul = copy.deepcopy(X)
    mul[0] *= Y[0][cat-1]
    mul[1] *= Y[0][cat-1]
    mul[2] *= Y[0][cat-1]

    scaled = (Y[0, cat-1] - min) / (max - min)
    scaled = scaled.reshape((1, scaled.shape[0], scaled.shape[1]))

    masked = copy.deepcopy(X)
    masked[0] *= mask
    masked[1] *= mask
    masked[2] *= mask

    output = np.zeros((3, img_Y.shape[0], img_Y.shape[1]))
    output[1] = (img_Y.cpu() * 2.)
    output[0] = (img_Y.cpu() * -2.) + 2.
    output[0, -20:, -20:] = 1.
    output[1, -20:, -20:] = 1.
    #output[0]
    
    #img_Y = np.array(img_Y.cpu() * 255, dtype=np.uint8)
    #output = cv.applyColorMap(img_Y, cv.COLORMAP_COOL)
    #output = cv.cvtColor(output, cv.COLOR_RGB2BGR)
    #output = output / 255
    #output = output.transpose((2, 0, 1))

    writer.add_image(f'output/{ds_train.categories[cat]}', output, epoch)
    writer.add_image(f'img_Y/{ds_train.categories[cat]}', img_Y.cpu().unsqueeze(0), epoch)
    writer.add_image(f'mul/{ds_train.categories[cat]}', mul, epoch)
    writer.add_image(f'scaled/{ds_train.categories[cat]}', scaled, epoch)
    writer.add_image(f'masked/{ds_train.categories[cat]}', masked, epoch)

if __name__ == '__main__':
    if os.path.isfile('model.sd'):
        with open('model.sd', 'rb') as f:
            model.load_state_dict(torch.load(f))

    x0, t0 = ds_val[10]
    writer.add_image('input', x0)
    model.eval()
    with torch.no_grad():
        y0 = model(x0.unsqueeze(0))
    vis_output(1, writer, x0, y0, epoch)#person
    vis_output(2, writer, x0, y0, epoch)#bicycle
    vis_output(3, writer, x0, y0, epoch)#car
    vis_output(4, writer, x0, y0, epoch)#cow
    writer.flush()

    while epoch < epochs:
        loss_train = train(model, train_loader, optimizer, criterion, batch_size, iterations=10)
        loss_val = val(model, val_loader, criterion, batch_size, iterations=5)
        writer.add_scalars("Loss", {'train': loss_train, 'val': loss_val}, epoch)
        acc_train = L.get_accuracy(model, val_loader, batch_size, iterations=5)
        acc_val = L.get_accuracy(model, val_loader, batch_size, iterations=5)
        writer.add_scalars("accuracy", {'train': acc_train, 'val': acc_val}, epoch)

        model.eval()
        with torch.no_grad():
            y0 = model(x0.unsqueeze(0))
        vis_output(1, writer, x0, y0, epoch)#person
        vis_output(2, writer, x0, y0, epoch)#bicycle
        vis_output(3, writer, x0, y0, epoch)#car
        vis_output(4, writer, x0, y0, epoch)#cow

        writer.flush()
        gc.collect()
        
        print(f'epoch {epoch} train {loss_train:0.5f} val {loss_val:0.5f} acc_train {(acc_train*100):0.3f}% acc_val {(acc_val*100):0.3f}%')

        with open('model.sd', 'wb') as f:
            torch.save(model.state_dict(), f)
        epoch += 1
    writer.close()
    