from __future__ import (absolute_import, division, print_function, unicode_literals)

import argparse
from operator import le
from typing import List
import os
import numpy as np
import torch
import glob
import torch.nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.autograd import Variable
import torchvision
import pathlib
from tqdm import tqdm
import matplotlib.pylab as plt
import tensorflow as tf
# import tensorflow_hub as hub
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from path import Path
import tensorflow_hub as hub
from word_detector import detect, prepare_img, sort_multiline
import tensorflow as tf
import numpy as np
# import pytorch as torch

def get_img_files(data_dir: Path) -> List[Path]:
    """Return all image files contained in a folder."""
    res = []
    for ext in ['*.png', '*.jpg','*.jpeg', '*.bmp']:
        res += Path(data_dir).files(ext)
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('../data/line'))
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=5)
    parser.add_argument('--theta', type=float, default=4)
    parser.add_argument('--min_area', type=int, default=400)
    parser.add_argument('--img_height', type=int, default=250)
    parsed = parser.parse_args()

    class ConvNet(nn.Module):
        def __init__(self,num_classes=6):
            super(ConvNet,self).__init__()
            
            #Output size after convolution filter
            #((w-f+2P)/s) +1
            
            #Input shape= (256,3,150,150)
            
            self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3,stride=1,padding=1)
            #Shape= (256,12,150,150)
            self.bn1=nn.BatchNorm2d(num_features=12)
            #Shape= (256,12,150,150)
            self.relu1=nn.ReLU()
            #Shape= (256,12,150,150)
            
            self.pool=nn.MaxPool2d(kernel_size=2)
            #Reduce the image size be factor 2
            #Shape= (256,12,75,75)
            
            
            self.conv2=nn.Conv2d(in_channels=12,out_channels=20,kernel_size=3,stride=1,padding=1)
            #Shape= (256,20,75,75)
            self.relu2=nn.ReLU()
            #Shape= (256,20,75,75)
            
            
            
            self.conv3=nn.Conv2d(in_channels=20,out_channels=32,kernel_size=3,stride=1,padding=1)
            #Shape= (256,32,75,75)
            self.bn3=nn.BatchNorm2d(num_features=32)
            #Shape= (256,32,75,75)
            self.relu3=nn.ReLU()
            #Shape= (256,32,75,75)
            
            
            self.fc=nn.Linear(in_features=112 * 112 * 32,out_features=num_classes)

        def forward(self,input):
            output=self.conv1(input)
            output=self.bn1(output)
            output=self.relu1(output)
                
            output=self.pool(output)
                
            output=self.conv2(output)
            output=self.relu2(output)
                
            output=self.conv3(output)
            output=self.bn3(output)
            output=self.relu3(output)
                
                
                #Above output will be in matrix form, with shape (256,32,75,75)
                
            output=output.view(-1,32*112*112)
                
                
            output=self.fc(output)
                
            return output

            #Feed forwad function
    # device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emoji_model = tf.keras.models.load_model('../A_model26.h5',custom_objects={'KerasLayer':hub.KerasLayer})

    for fn_img in get_img_files(parsed.data):
        # print(f'Processing file {fn_img}')

        # load image and process it
        mx=0
        word=""
        for l_th in [130]: # change threshold value acc. to your image or give multiple thresholds
            img = prepare_img(cv2.imread(fn_img), parsed.img_height)
            ret, img = cv2.threshold(img, l_th, 255, cv2.THRESH_BINARY)
            detections = detect(img,
                                kernel_size=parsed.kernel_size,
                                sigma=parsed.sigma,
                                theta=parsed.theta,
                                min_area=parsed.min_area)

            # sort detections: cluster into lines, then sort each line
            lines = sort_multiline(detections)

            # plot results
            plt.imshow(img, cmap='gray')
            num_colors = 7
            colors = plt.cm.get_cmap('rainbow', num_colors)
            nmx=0
            nword=""
            for line_idx, line in enumerate(lines):
                # print(line)
                for word_idx, det in enumerate(line):
                    ret, img = cv2.threshold(det.img, 150, 255, cv2.THRESH_BINARY)

                    img = cv2.resize(img,(28,28))
                    kernel = np.ones((2,2), np.uint8)
                    img = cv2.erode(img, kernel, iterations=2)
                    img = cv2.dilate(img, kernel, iterations=1)
                    
                    img=img.reshape(28,28,1)
                    img=np.stack((img,))
                    # print(img.shape)
                    emoji_ped = emoji_model.predict(img)
                    emoji_index = np.argmax(emoji_ped)
                    nmx+=emoji_ped[0][emoji_index]
                    mp=['A','D','E','G','H','J','K','M','N','P','R','S','W','X', 'Z', 'a', 'b', 'd', 'g', '1', '2', '3', '4', '5', '6', '7']
                    nword+=mp[emoji_index]

                    xs = [det.bbox.x, det.bbox.x, det.bbox.x + det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                    ys = [det.bbox.y, det.bbox.y + det.bbox.h, det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                    plt.plot(xs, ys, c=colors(line_idx % num_colors))
                    plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')

            if(mx<nmx):
                mx=nmx
                word=nword
        print(word)
        plt.show()


if __name__ == '__main__':
    main()
