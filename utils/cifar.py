import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import random
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms

from .randaugment import RandAugmentMC

logger = logging.getLogger(__name__)

cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)
cifar100_mean = (0.5071, 0.4867, 0.4408)
cifar100_std = (0.2675, 0.2565, 0.2761)
normal_mean = (0.5, 0.5, 0.5)
normal_std = (0.5, 0.5, 0.5)
img_size = 64

path_data = str(Path("data/tiles").resolve())
csv_path = {
    'train': os.path.join(path_data, "single_img_csv/upsample_train.csv"),
    'dev': os.path.join(path_data, "single_img_csv/validation.csv"),
}
print(csv_path)
path_train = os.path.join(path_data, "all_data")
path_validation = os.path.join(path_data, "all_data")
print(path_train)
DATA_DIR = path_train
NEGATIVE_DIR = path_train
SAVE_DIR = 'models/'

train_losses = []
dev_losses = []

image_classes = ['cored', 'diffuse', 'CAA']


def get_cifar10(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    base_dataset = datasets.CIFAR10(root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR10SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR10SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar10_mean, std=cifar10_std))

    test_dataset = datasets.CIFAR10(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_cifar100(args, root):
    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=32,
                              padding=int(32 * 0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar100_mean, std=cifar100_std)])

    base_dataset = datasets.CIFAR100(
        root, train=True, download=True)

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.targets)

    train_labeled_dataset = CIFAR100SSL(
        root, train_labeled_idxs, train=True,
        transform=transform_labeled)

    train_unlabeled_dataset = CIFAR100SSL(
        root, train_unlabeled_idxs, train=True,
        transform=TransformFixMatch(mean=cifar100_mean, std=cifar100_std))

    test_dataset = datasets.CIFAR100(
        root, train=False, transform=transform_val, download=False)

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def get_med_img(args, root):
    norm = np.load('utils/normalization.npy', allow_pickle=True).item()
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Resize(img_size),
            # transforms.RandomRotation(180),
            transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.2, hue=0.02),
            transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.9, 1.1), shear=10),
            transforms.ToTensor(),
            transforms.Normalize(normal_mean, normal_std)
        ]),
        'dev': transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(normal_mean, normal_std)
        ]),
    }

    base_dataset = MultilabelDataset(csv_path['train'], path_train, data_transforms['train'])

    train_labeled_idxs, train_unlabeled_idxs = x_u_split(
        args, base_dataset.l)

    train_labeled_dataset = medImgSSL(
        csv_path['train'], path_train, train_labeled_idxs,
        transform=data_transforms['train'])

    train_unlabeled_dataset = medImgSSL(
        csv_path['train'], path_train, train_unlabeled_idxs,
        transform=TransformFixMatch(mean=norm['mean'], std=norm['std']))

    test_dataset = MultilabelDataset(csv_path['dev'], path_validation, data_transforms['dev'])

    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def x_u_split(args, labels):
    a = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    label_per_class = args.num_labeled // len(a)
    labels = np.array(labels).tolist()
    b = [[],[],[]]
    for i in range(len(labels)):
        label = labels[i]
        if label == a[0]:
            b[0].append(i)
        elif label == a[1]:
            b[1].append(i)
        elif label == a[2]:
            b[2].append(i)
    print(len(b[0]), len(b[1]), len(b[2]))
    labeled_idx = []
    # unlabeled data: all data (https://github.com/kekmodel/FixMatch-pytorch/issues/10)
    unlabeled_idx = np.array(range(len(labels)))
    # for i in range(len(a)):
    #     idx = np.where(labels == np.array(a[i]))[0]
    #     print(idx)
    #     idx = np.random.choice(idx, label_per_class, False)
    #     labeled_idx.extend(idx)
    for bb in b:
        idx = random.sample(range(len(bb)), label_per_class)
        for i in idx:
            labeled_idx.append(bb[i])
    labeled_idx = np.array(labeled_idx)
    assert len(labeled_idx) == args.num_labeled

    if args.expand_labels or args.num_labeled < args.batch_size:
        num_expand_x = math.ceil(
            args.batch_size * args.eval_step / args.num_labeled)
        labeled_idx = np.hstack([labeled_idx for _ in range(num_expand_x)])
    np.random.shuffle(labeled_idx)
    return labeled_idx, unlabeled_idx


class TransformFixMatch(object):
    def __init__(self, mean, std):
        self.weak = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(72),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size * 0.125),
                                  padding_mode='reflect')])
        self.strong = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(72),
            transforms.RandomCrop(size=img_size,
                                  padding=int(img_size * 0.125),
                                  padding_mode='reflect'),
            RandAugmentMC(n=2, m=10)])
        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return self.normalize(weak), self.normalize(strong)


class CIFAR10SSL(datasets.CIFAR10):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class CIFAR100SSL(datasets.CIFAR100):
    def __init__(self, root, indexs, train=True,
                 transform=None, target_transform=None,
                 download=False):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,
                         download=download)
        if indexs is not None:
            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class MultilabelDataset(Dataset):
    def __init__(self, csv_path, img_path, transform=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data_info = pd.read_csv(csv_path)
        self.img_path = img_path
        self.transform = transform
        c = torch.Tensor(self.data_info.loc[:, 'cored'])
        d = torch.Tensor(self.data_info.loc[:, 'diffuse'])
        a = torch.Tensor(self.data_info.loc[:, 'CAA'])
        c = c.view(c.shape[0], 1)
        d = d.view(d.shape[0], 1)
        a = a.view(a.shape[0], 1)
        self.raw_labels = torch.cat([c, d, a], dim=1)
        self.l = (torch.cat([c, d, a], dim=1) > 1).type(torch.FloatTensor)
        self.indexes = self.data_info.index


    def __getitem__(self, index):
        idx = self.indexes[index]
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.l[idx]
        max_probs, single_image_label = torch.max(single_image_label, dim=-1)
        # raw_label = self.raw_labels[index]
        # Get image name from the pandas df
        single_image_name = str(self.data_info.loc[idx,'imagename'])
        # Open image
        try:
            img_as_img = Image.open(os.path.join(self.img_path, single_image_name))
        except:
            img_as_img = Image.open(os.path.join(NEGATIVE_DIR, single_image_name))
        # Transform image to tensor
        if self.transform is not None:
            img_as_img = self.transform(img_as_img)
        return img_as_img, int(single_image_label)  # , raw_label, single_image_name

    def __len__(self):
        return len(self.indexes)


class medImgSSL(MultilabelDataset):
    def __init__(self, csv_path, path_train, indexs, transform=None):
        super().__init__(csv_path, path_train, transform=transform)
        if indexs is not None:
            self.indexes = indexs
            # self.targets = np.array(self.targets)[indexs]

    # def __getitem__(self, index):
    #     img, target = self.data[index], self.targets[index]
    #     img = Image.fromarray(img)

    #     if self.transform is not None:
    #         img = self.transform(img)

    #     if self.target_transform is not None:
    #         target = self.target_transform(target)

    #     return img, target


DATASET_GETTERS = {'cifar10': get_cifar10,
                   'cifar100': get_cifar100,
                   'med-img': get_med_img}
