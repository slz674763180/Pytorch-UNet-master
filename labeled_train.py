import random
import numpy as np
import torch
import torch.nn as nn
import scipy.misc as mi
from PIL import Image
from torch import optim
# from dice_loss import *
import scipy.io as io
from unet import UNet
from dice_loss import *
# from test import *

# weight = [1, 10]
# w = torch.from_numpy(np.array(weight, dtype=np.float32))

# class_weight = Variable(torch.FloatTensor([1, 10])) # 这里正例比较少，因此权重要大一些
# target = Variable(torch.FloatTensor(1,250, 250).random_(2))
# weight = class_weight[target.long()] # (3, 4)

# criterion = nn.CrossEntropyLoss(weight=w).cuda()
cc = nn.BCELoss().cuda()
ccc = MulticlassDiceLoss()
dd = DiceLoss()
labeled_num = 150
# criterion1 = SegmentationLosses(weight=weight, cuda=True).build_loss(mode='ce')

def get_data():
    imgs = []
    masks = []
    data_dir = 'data'
    i = 0
    while i < labeled_num:
        img = Image.open(data_dir + '/train' + '/' + str(i) + '.png')
        img = np.array(img, dtype=np.float32) / 255
        img = img[np.newaxis, np.newaxis, :, :]

        mask = io.loadmat(data_dir + '/train_masks' + '/' + str(i) + '.mat')['img']
        mask[(mask != 1) * (mask != 2) * (mask != 3) * (mask != 4) * (mask != 5)] = 0
        mask = np.array(mask, dtype=np.float32)
        mask = mask[np.newaxis, :, :]

        imgs.append(img)
        masks.append(mask)
        i += 1
    return imgs, masks


def get_test(i):
    img = Image.open('data/test' + '/' + str(i) + '.png')
    img = np.array(img, dtype=np.float32) / 255
    img = img[np.newaxis, np.newaxis, :, :]

    mask = io.loadmat('data/test_masks' + '/' + str(i) + '.mat')['img']
    mask[(mask != 1) * (mask != 2) * (mask != 3) * (mask != 4) * (mask != 5)] = 0
    mask = np.array(mask, dtype=np.float32)

    mask = mask[np.newaxis, :, :]
    return img, mask


def compared(masks_preds, y, j):
    masks_preds = np.array(masks_preds.cpu().detach().numpy())
    masks_preds = np.squeeze(masks_preds)
    masks_preds[masks_preds>0.5]=1
    masks_preds[masks_preds <= 0.5] = 0
    result = masks_preds
    # result = np.argmax(masks_preds, axis=0)
    y = np.array(y)
    y = np.squeeze(y)
    y[y!=0]=1
    a = [0,0,0,0,0]
    # result = masks_preds[4] * 5
    # for i in range(4):
    #     result += masks_preds[i] * (i+1)

    mi.imsave('data/result_labeled/' + str(j) + '.png', result)
    cont = 0
    total = 0
    ha = 0
    for pp in range(len(y[0])):
        for qq in range(len(y[1])):
            if y[pp][qq] == result[pp][qq]:
                ha += 1
            if y[pp][qq] != 0:
                a[int(y[pp][qq])-1]=1
                total += 1
                if result[pp][qq] == y[pp][qq]:
                    cont += 1
            if result[pp][qq] != 0:
                total += 1
    mean = 2 * cont / total
    return a, mean , ha/250/250


def compared1(masks_preds, y, j):
    masks_preds = np.array(masks_preds.cpu().detach().numpy())
    masks_preds = np.squeeze(masks_preds)
    result = np.argmax(masks_preds, axis=0)
    y = np.array(y)
    y = np.squeeze(y)
    a = [0,0,0,0,0]
    # result = masks_preds[5] * 5
    # for i in range(5):
    #     result += masks_preds[i] * i

    # mi.imsave('data/result_labeled/' + str(j) + '.png', result)
    cont = 0
    total = 250*250
    for pp in range(len(y[0])):
        for qq in range(len(y[1])):
            if y[pp][qq] == result[pp][qq]:
                if y[pp][qq]!=0:
                    a[int(y[pp][qq])-1] = 1
                cont += 1
    mean = cont / total
    return a, mean

def domain():
    net = UNet(n_channels=1, n_classes=6)
    net.load_state_dict(torch.load('weight/labeled.pth'))
    net.cuda()
    i = 1000
    means = 0
    while i < 1321:
        x, y = get_test(i)
        imgs = torch.from_numpy(x)
        imgs = imgs.cuda()
        masks_preds = net(imgs)
        masks_preds = F.softmax(masks_preds, 1)
        mean = compared(masks_preds, y, i)
        print('accuracy of {} ='.format(i), mean)
        means += mean
        i += 1
    print('accuracy=', means / 321)


def train(model, imgs, masks, optimizer, epoch, step_counter):
    model.train()
    j = 20
    for i in range(len(imgs)):
        # if j==30:
        #     j=20
        # i = j
        # j += 1
        img = imgs[i]
        mask = masks[i]
        mask[mask!=0]=1
        # maskkk = []
        # for jj in range(6):
        #     if jj == 0:
        #         continue
        #     label = mask.copy()
        #     label[mask == jj] = 1
        #     label[mask != jj] = 0
        #     maskkk.append(label)
        # maskkk = np.array(maskkk).squeeze()
        # maskkk = maskkk[np.newaxis, :, :, :]

        # aabb = torch.from_numpy(np.array(maskkk, dtype=np.float32))
        image = torch.from_numpy(np.array(img, dtype=np.float32))
        maskk = torch.from_numpy(np.array(mask, dtype=np.float32))
        input_var = image.cuda()
        target_var = maskk.cuda()
        # aabb = aabb.cuda()
        model_out = model(input_var)
        a = np.array(model_out.cpu().detach().numpy())
        # b = np.array(mask.cpu().detach().numpy())


        # masks_preds = F.softmax(model_out, 1)
        c, mean, acc = compared(model_out, mask, i)
        # c, acc = compared1(masks_preds, mask, i)

        loss = dd(model_out, target_var)
        # loss = ccc(model_out, aabb)
        # loss = loss1 + 2*loss2
        print('i={},sum of target={} dice={:.3f} , acc={:.3f} of {}, loss={:.3f}'.format(i,mask.sum(),mean, acc, c, loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_counter += 1
        # print("{} - {}, loss: {:.3f}".format(step_counter, len(imgs), float(loss)))
    torch.save(model.state_dict(), "weight/labeled.pth")
    return step_counter





# domain()
model = UNet(n_channels=1, n_classes=1)
# model.load_state_dict(torch.load('weight/labeled.pth'))
model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

imgs, masks = get_data()
# cc = list(zip(imgs, masks))
# random.shuffle(cc)
# imgs[:], masks[:] = zip(*cc)

for epoch in range(50):
    train(model, imgs, masks, optimizer, epoch, 0)
    print('epoch=', epoch + 1)
    # domain()
