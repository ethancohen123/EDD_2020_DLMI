#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 10:24:45 2020

@author: shariba
"""

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

import numpy as np
from utils.utils import batch
from utils.data_Loader import get_imgs_and_masks, split_train_val
from utils.visualize import inputImageViz

from tqdm import tqdm


class ConvBlock(nn.Module):
    """Basic convolutional block."""

    def __init__(self, in_channels, out_channels, norm='batch'):
        super().__init__()
        # choice of padding=1 keeps
        # feature map dimensions identical
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        if norm == 'batch':
            self.bn = nn.BatchNorm2d(out_channels)
        elif norm == 'group':
            num_groups = out_channels // 8
            self.bn = nn.GroupNorm(num_groups, out_channels)
        elif norm is None:
            self.bn = nn.Identity()
        else:
            raise TypeError(
                "Wrong type of normalization layer provided for ConvBlock")
        self.activation = nn.ReLU()

    def forward(self, x: Tensor):
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class _DownBlock(nn.Module):
    """Contracting path segment.

    Downsamples using MaxPooling then applies ConvBlock.
    """

    def __init__(self, in_channels, out_channels, n_convs=2):
        super().__init__()
        layers = [
            ConvBlock(in_channels, out_channels)
        ] + [
            ConvBlock(out_channels, out_channels)
            for _ in range(n_convs-1)
        ]
        # maxpooling over patches of size 2
        self.mp = nn.MaxPool2d(2)
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.mp(x)
        x = self.conv(x)
        return x


class _UpBlock(nn.Module):
    """Expansive path segment.

    Applies `~ConvBlock`, then upsampling deconvolution.
    """

    def __init__(self, in_channels, out_channels, n_convs=2, n_connect=2):
        """

        Parameters
        ----------
        n_connect : int
            Multiplicator for the number of input for the 1st convblock after
            the upsampling convolution (useful for skip connections).
        """
        super().__init__()
        layers = [
            # expects multiple of channels
            ConvBlock(n_connect * in_channels, in_channels)
        ] + [
            ConvBlock(in_channels, in_channels)
            for _ in range(n_convs-1)
        ]
        self.conv = nn.Sequential(*layers)
        # counts as one convolution
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels,
                                         2, stride=2)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        z = torch.cat((skip, x), dim=1)
        z = self.conv(z)
        out = self.upconv(z)  # deconvolve
        return out


class UNet(nn.Module):
    """The U-Net architecture.

    See https://arxiv.org/pdf/1505.04597.pdf
    """

    def __init__(self, num_channels: int = 3, num_classes: int = 5):
        """Initialize a U-Net.

        Parameters
        ----------
        num_channels : int
            Number of input channels.
        num_classes : int
            Number of output classes.
        """
        super().__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes

        self.in_conv = nn.Sequential(
            ConvBlock(num_channels, 64),
            ConvBlock(64, 64)
        )

        self.down1 = _DownBlock(64, 128)
        self.down2 = _DownBlock(128, 256)
        self.down3 = _DownBlock(256, 512)

        self.center = nn.Sequential(
            _DownBlock(512, 1024),
            nn.ConvTranspose2d(1024, 512, 2, stride=2)  # upscale
        )

        # reminder: convolves then upsamples
        self.up1 = _UpBlock(512, 256)
        self.up2 = _UpBlock(256, 128)
        self.up3 = _UpBlock(128, 64)

        self.out_conv = nn.Sequential(
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, num_classes, kernel_size=1, padding=0)
        )

    def forward(self, x: Tensor):
        x1 = self.in_conv(x)  # 64 * 1. * 1. ie 224
        x2 = self.down1(x1)  # 128 * 1/2 * 1/2
        x3 = self.down2(x2)  # 256 * 1/4 * 1/4
        x4 = self.down3(x3)  # 512 * 1/8 * 1/8
        x = self.center(x4)  # 512 * 1/8 * 1/8 ie 28
        x = self.up1(x, x4)  # 256 * 1/4 * 1/4 56
        x = self.up2(x, x3)  # 128 * 1/2 * 1/2 112
        x = self.up3(x, x2)
        z = torch.cat((x1, x), dim=1)
        out = self.out_conv(z)
        return out


def get_args():
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch_size', dest='batch_size', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.1,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    # input settings
    parser.add_option('-m', '--mask_Dir', dest='masks', default='sampleDataLoader_semantic/data/train_masks/',
                      type='str', help='base direcotry for masks ')
    parser.add_option('-i', '--image_Dir', dest='images', default='sampleDataLoader_semantic/data/train_images/',
                      type='str', help='base direcotry for images ')
    parser.add_option('-w', '--maxWidth', dest='maxWidth', default=512,
                      type='int', help='max image width (to be resized if not same)')
    parser.add_option('-z', '--maxHeight', dest='maxHeight', default=512,
                      type='int', help='max image height (to be resized if not same)')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')
    parser.add_option('-v', '--val_percent', dest='valpercent', type='float',
                      default=0.1, help='validation percentage for splitting')
    parser.add_option('-n', '--n_classes', dest='classes', default=5,
                      type='int', help='number of class labels')

    (options, args) = parser.parse_args()
    return options


device = "cuda" if torch.cuda.is_available() else "cpu"


def dice_loss(input, target):
    smooth = 1.
    target = target.float()
    input = input.float()
    input_flat = input.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)
    intersection = (input_flat * target_flat).sum()
    return 1 - ((2. * intersection + smooth) /
                (input_flat.pow(2).sum() + target_flat.pow(2).sum() + smooth))


def mean_dice_loss(input, target):
    # print('T', target.shape)
    channels = list(range(target.shape[1]))
    loss = 0
    for channel in channels:
        dice = dice_loss(input[:, channel, ...],
                         target[:, channel, ...])
        loss += dice

    return loss / len(channels)


class EDDDataset(Dataset):

    def __init__(self, iddataset, dir_img, dir_mask, scale, maxWidth, maxHeight, n_classes, transform=None):
        imgs_normalized, masks_switched = get_imgs_and_masks(
            iddataset, dir_img, dir_mask, scale, maxWidth, maxHeight, n_classes)

        self.imgs_normalized = imgs_normalized
        self.masks_switched = masks_switched

    def __len__(self):
        return len(self.imgs_normalized)

    def __getitem__(self, idx):
        return self.imgs_normalized[idx], self.masks_switched[idx]


if __name__ == '__main__':
    args = get_args()
    dir_img = args.images
    dir_mask = args.masks
    dir_checkpoint = 'checkpoints/'

    val_percent = args.valpercent
    scale = args.scale
    maxWidth = args.maxWidth
    maxHeight = args.maxHeight
    n_classes = args.classes

    batch_size = args.batch_size

    iddataset = split_train_val(dir_img, val_percent)

    dataset_train = EDDDataset(
        iddataset['train'], dir_img, dir_mask, scale, maxWidth, maxHeight, n_classes)
    dataset_val = EDDDataset(
        iddataset['val'], dir_img, dir_mask, scale, maxWidth, maxHeight, n_classes)

    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True)
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, shuffle=False)

    model = UNet()
    criterion = mean_dice_loss

    for epoch in range(args.epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, args.epochs))
        model.train()
        model.to(device)
        for i, batch in tqdm(enumerate(dataloader_train)):
            imgs, true_masks = batch
            # inputImageViz(imgs, true_masks)

            imgs = imgs.float().to(device)
            masks_gd = true_masks.float().to(device)

            masks_pred = model(imgs)
            # inputImageViz(imgs.detach().cpu().numpy(),
            #               masks_pred.detach().cpu().numpy())
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
            # comment this if required

            # pred = torch.argmax(output, dim=1, keepdim=True)
            loss = criterion(masks_pred, masks_gd)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('Loss: ', loss.item())
