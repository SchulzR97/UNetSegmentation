import torch
import numpy as np

class conv_block(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_c)
        self.conv2 = torch.nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_c)
        self.relu = torch.nn.PReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class encoder_block(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = torch.nn.MaxPool2d((2, 2))
    
    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p

class decoder_block(torch.nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = torch.nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)
    
    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x

class UNet(torch.nn.Module):
    def __init__(self, device = 'cpu', num_categories = None, scale_size = 1):
        super().__init__()

        self.num_categories = 90 if num_categories is None else num_categories

        """ Encoder """
        self.e1 = encoder_block(3, 64 * scale_size)
        self.e2 = encoder_block(64 * scale_size, 128 * scale_size)
        self.e3 = encoder_block(128 * scale_size, 256 * scale_size)
        self.e4 = encoder_block(256 * scale_size, 512 * scale_size)
        """ Bottleneck """
        self.b = conv_block(512 * scale_size, 1024 * scale_size)
        """ Decoder """
        self.d1 = decoder_block(1024 * scale_size, 512 * scale_size)
        self.d2 = decoder_block(512 * scale_size, 256 * scale_size)
        self.d3 = decoder_block(256 * scale_size, 128 * scale_size)
        self.d4 = decoder_block(128 * scale_size, 64 * scale_size)
        """ Classifier """
        self.outputs = torch.nn.Conv2d(64 * scale_size, self.num_categories, kernel_size=1, padding=0)

        self.sigmoid = torch.nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)

        self.device = device
        self.to(device)
    
    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return self.sigmoid(outputs)
        #return self.softmax(outputs)

class SegNet(torch.nn.Module):
    def __init__(self, size, device = 'mps'):
        super().__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(9,9), stride = 1, padding=4)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(7,7), padding=3)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=(5,5), stride=2, padding=2, return_indices=True)

        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(5,5), padding=2)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1, return_indices=True)

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=(3,3), padding=1)
        self.maxpool4 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1, return_indices=True)

        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=(3,3), padding=1)
        self.maxpool5 = torch.nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=1, return_indices=True)

        self.conv5T = torch.nn.Conv2d(512, 256, kernel_size=(3,3), padding=1)
        self.unpool5 = torch.nn.MaxUnpool2d(kernel_size=(4,4), stride=2, padding=1)

        self.conv4T = torch.nn.Conv2d(256, 128, kernel_size=(3,3), padding=1)
        self.unpool4 = torch.nn.MaxUnpool2d(kernel_size=(3,3), stride=2, padding=1)

        self.conv3T = torch.nn.Conv2d(128, 64, kernel_size=(7,7), padding=1)
        self.unpool3 = torch.nn.MaxUnpool2d(kernel_size=(5,5), stride=2, padding=0)

        self.conv2T = torch.nn.Conv2d(64, 90, kernel_size=(2,2), padding=0)
        self.unpool2 = torch.nn.MaxUnpool2d(kernel_size=(7,7), stride=2, padding=2)

        #self.conv5T = torch.nn.ConvTranspose2d(512, 256, kernel_size=(2,2), stride=2, padding=0)
        #self.conv4T = torch.nn.ConvTranspose2d(256, 128, kernel_size=(2,2), stride=2, padding=0)
        #self.conv3T = torch.nn.ConvTranspose2d(128, 64, kernel_size=(4,4), stride=2, padding=1)
        #self.conv2T = torch.nn.ConvTranspose2d(64, 32, kernel_size=(2,2), stride=2, padding=3)
        #self.conv1T = torch.nn.ConvTranspose2d(32, 90, kernel_size=(1,1), stride=1, padding=0)
        
        #self.conv1T = torch.nn.Conv2d(32, 90, kernel_size=(6,6), stride=1, padding=0)

        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(384, 190)
        #self.fc2 = torch.nn.Linear(4000, 4000)
        self.fc3 = torch.nn.Linear(190, 90*size[0]*size[1])
        
        self.relu = torch.nn.LeakyReLU()

        self.unflatten = torch.nn.Unflatten(dim=1, unflattened_size=(90, size[0], size[1]))

        self.to(device)

    def forward(self, X):
        Y = X

        # encoder
        #print(Y.shape)
        Y = self.relu(self.conv1(Y))
        #print(Y.shape)
        
        Y, indices2 = self.maxpool2(self.conv2(Y))
        Y = self.relu(Y)
        #print(Y.shape)
        
        Y, indices3 = self.maxpool3(self.conv3(Y))
        Y = self.relu(Y)
        #print(Y.shape)

        Y, indices4 = self.maxpool4(self.conv4(Y))
        Y = self.relu(Y)
        #print(Y.shape)

        Y, indices5 = self.maxpool5(self.conv5(Y))
        Y = self.relu(Y)
        #print(Y.shape)

        # decoder
        Y = self.unpool5(Y, indices5)
        Y = self.relu(self.conv5T(Y))
        #print(Y.shape)

        Y = self.unpool4(Y, indices4)
        Y = self.relu(self.conv4T(Y))
        #print(Y.shape)

        Y = self.unpool3(Y, indices3)
        Y = self.relu(self.conv3T(Y))
        #print(Y.shape)

        Y = self.unpool2(Y, indices2)
        Y = self.relu(self.conv2T(Y))
        #print(Y.shape)

        return Y