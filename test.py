import cv2
import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from torch import optim

from unet import UNet







def saveResult(save_path, result, o):
    num = np.shape(result)[0]
    for b in range(int(num)):
        img = result[b]
        img = img.reshape((128, 128))
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        img = img.detach().cpu().numpy()*255
        cv2.imwrite(os.path.join(save_path, str(o) + ".png"), img)
        o += 1





def compare(path, result,o,mean):
    while o < 578:
        mask = Image.open(path + '/' + str(o) + '.jpg')
        mask = mask.resize((128, 128), Image.ANTIALIAS)
        mask = np.array(mask) / 255
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        img = Image.open(result + '/' + str(o) + '.png')
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img) / 255
        img[img >= 0.5] = 1
        img[img < 0.5] = 0
        cont = 0
        total = 0



        for pp in range(128):
            for qq in range(128):
                if mask[pp][qq] == 1:
                    total += 1
                    if img[pp][qq] == 1:
                        cont += 1
                if img[pp][qq] == 1:
                    total += 1
        mean += 2 * cont / total
        o += 1
    return mean


name = 'teacher'


def domain():
    net = UNet(n_channels=1, n_classes=1)
    net.load_state_dict(torch.load('weight/'+name+'.pth'))
    # net.load_state_dict(torch.load('weight/teacher.pth'))
    net.cuda()
    o = 200
    i = 200
    while (i < 578):
        x = get_data(i)
        imgs = torch.from_numpy(x)
        imgs = imgs.cuda()
        masks_preds = net(imgs)
        # saveResult('data/result_labeled', masks_preds)
        saveResult('data/result_'+name, masks_preds, o)
    mean = 0
    o = 200
    mean = compare('data/unlabel_masks', 'data/result_'+name, o, mean)
    print('accuracy=', mean / 378)


if __name__ == '__main__':
    domain()
