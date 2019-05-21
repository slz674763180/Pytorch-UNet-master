import random

import numpy as np
import cv2

from get_data import *
import torch
import torch.nn as nn
import torchvision.transforms as tfs
from PIL import Image
from torch import optim
from dice_loss import *
from unet import UNet

criterion = DiceLoss()
criteriond = nn.BCELoss()
lr = 0.1
epochs = 2
batch_size = 4
labeled_num = 500


def get_data():
    dataset = []
    data_dir = 'data'
    i = 0
    while i < labeled_num:
        img = Image.open(data_dir + '/train' + '/' + str(i) + '.png')
        mask = Image.open(data_dir + '/train_masks' + '/' + str(i) + '.png')
        img = img.resize((128, 128), Image.ANTIALIAS)
        mask = mask.resize((128, 128), Image.ANTIALIAS)

        img = np.array(img, dtype=np.float32) / 255
        mask = np.array(mask, dtype=np.float32) / 255
        img = img[np.newaxis, np.newaxis, :, :]
        mask = mask[np.newaxis, np.newaxis, :, :]
        dataset.append(data(img, mask, 1))
        i += 1
    return dataset

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    # If no rotation center is specified, the center of the image is set as the rotation center
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated

def get_unlabel(dataset):
    data_dir = 'data'
    i = labeled_num
    while i < 78:
        img = Image.open(data_dir + '/unlabel' + '/' + str(i) + '.png')
        img = img.resize((128, 128), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32) / 255
        img = img[np.newaxis, np.newaxis, :, :]


        # im_aug = tfs.RandomRotation(5, )(img)
        # im_aug = np.array(im_aug, dtype=np.float32) / 255
        # im_aug = im_aug[np.newaxis, np.newaxis, :, :]
        dataset.append(data(img, img, 0))
        # dataset.append(data(im_aug, im_aug, 0))
        i += 1
    return dataset


def update_ema_variables(model, ema_model, alpha, global_step):
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)


def GaussianNoise(src, means, sigma):
    NoiseImg = src
    rows = NoiseImg.shape[1]
    cols = NoiseImg.shape[2]
    for i in range(rows):
        for j in range(cols):
            NoiseImg[:, i, j] = NoiseImg[:, i, j] + random.gauss(means, sigma)
            if NoiseImg[:, i, j] < 0:
                NoiseImg[:, i, j] = 0
            elif NoiseImg[:, i, j] > 1:
                NoiseImg[:, i, j] = 1
    return NoiseImg


def dice_score(x,y):
    x = np.rint(x)
    y = np.rint(y)
    return 2*(x*y).sum()/(x.sum()+y.sum())


def compared(s_preds, t_preds, target, labeled):
    s_preds = np.array(s_preds.cpu().detach().numpy())
    s_preds = np.squeeze(s_preds)

    t_preds = np.array(t_preds.cpu().detach().numpy())
    t_preds = np.squeeze(t_preds)

    target = np.array(target.cpu().detach().numpy())
    target = np.squeeze(target)
    s_mean = 0
    t_mean = 0
    if labeled==1:
        s_mean = dice_score(s_preds,target)
        t_mean = dice_score(t_preds,target)
    return s_mean,t_mean


def train_mt(model, ema_model, train_loader, optimizer, epoch, step_counter):
    alpha = 0.999
    model.train()
    ema_model.train()
    i = 0
    epoch_loss = 0

    # image0s = []
    # image1s = []
    # masks = []
    # jj = 0
    for input in train_loader:
        # if jj < 4:
        #     image = input.image
        #     image0 = image.copy()
        #     image1 = image.copy()
        #     mask = input.mask
        #     if i > 200:
        #         image0 = GaussianNoise(image0, 0, 0.1)
        #         image1 = GaussianNoise(image1, 0, 0.1)
        #     image0s.append(image0)
        #     image1s.append(image1)
        #     masks.append(mask)
        #     jj += 1
        #     i += 1
        #     if jj < 4:
        #         continue
        image0 = input.image.copy()
        image1 = input.image.copy()
        if input.labeled==0:
            image0 = GaussianNoise(image0, 0, 0.1)
            image1 = GaussianNoise(image1, 0, 0.1)

        image0 = torch.from_numpy(np.array(image0, dtype=np.float32))
        image1 = torch.from_numpy(np.array(image1, dtype=np.float32))
        mask = torch.from_numpy(np.array(input.mask, dtype=np.float32))

        labeled = input.labeled

        input_var = image0.cuda()
        ema_input_var = image1.cuda()

        target_var = mask.cuda()

        model_out = model(input_var)
        ema_model_out = ema_model(ema_input_var)

        s_mean, t_mean = compared(model_out, ema_model_out, target_var, labeled)


        model_out = model_out.view(-1)
        ema_model_out = ema_model_out.view(-1)
        target_var = target_var.view(-1)



        β = np.exp(-5 * (1 - step_counter / 100) ** 2)

        if labeled == 1:
            loss1 = criterion(model_out, target_var)
            loss2 = criterion(model_out, ema_model_out)

            loss = loss1 + float(min(β, 1)) * loss2
        else:
            loss = criterion(model_out, ema_model_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_counter += 1
        epoch_loss += loss
        update_ema_variables(model, ema_model, alpha, step_counter)
        print("{} - {}, s_dice = {:.2f}, t_dice = {:.2f}, loss: {:.2f}".format(step_counter, len(train_loader), s_mean, t_mean, float(loss)))

        #
        # image0s.clear()
        # image1s.clear()
        # masks.clear()
        # jj = 0

    torch.save(model.state_dict(), "weight/student.pth")
    torch.save(ema_model.state_dict(), "weight/teacher.pth")
    return step_counter


if __name__ == '__main__':
    model = UNet(n_channels=1, n_classes=1)
    # model.load_state_dict(torch.load('weight/student.pth'))
    model.cuda()
    ema_model = UNet(n_channels=1, n_classes=1)
    # ema_model.load_state_dict(torch.load('weight/teacher.pth'))
    ema_model.cuda()

    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0005)

    dataset = get_data()
    dataset = get_unlabel(dataset)

    for param in ema_model.parameters():
        param.detach_()

    random.shuffle(dataset)

    for epoch in range(10):
        train_mt(model, ema_model, dataset, optimizer, epoch, 0)
