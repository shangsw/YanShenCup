#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:00:58 2019

@author: ssw
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
import pickle
import glob
import h5py
import time

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')
def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions=None, is_valid_file=None):
    #return the path and labels according class_to_idx
    #由于pytorch自带ImageFolder返回随机的类标，因此这里传入class_to_idx字典，记录的是文件夹名(类名)和指定类标
    #class_to_idx: {'a':0, 'b':1, 'c':2}
    #return: images [(path1, index1),(path2, index2),...]
    images = []
    dir = os.path.expanduser(dir)
    if not ((extensions is None) ^ (is_valid_file is None)):
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue
        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images
    
class DataFolder(Dataset):
    #making dataset from images in folders, and with class correspond to index
    def __init__(self, root, class_to_idx, extensions=IMG_EXTENSIONS, 
                 transform=None, target_transform=None, 
                 is_valid_file=None, pre_load=False):
        self.transform = transform
        self.target_transform = target_transform
        self.root = root
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = self._pil_loader
        self.extensions = extensions

        self.classes = sorted(class_to_idx)
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.expand_samples = self.samples    #可能扩展进来新数据的情况，见expand函数
        self.data = None        
        if pre_load:
            self._pre_load()

    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    def _pre_load(self):
        self.data = []
        for (path, target) in self.samples:
            self.data.append([self.loader(path), target, path])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target, fname) where target is class_index of the target class, fname is the image name of the sample
        """
        if self.data is not None:
            sample = self.data[index][0]
            target = self.data[index][1]
            fname = self.data[index][2].split('/')[-1]
        else:
            path, target = self.expand_samples[index]
            sample = self.loader(path)
            fname = path.split('/')[-1]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, fname

    def __len__(self):
        return len(self.expand_samples)
    
    def expand(self, expand_list=[], temporarily=False):
        #当传入的列表中含有数据时，将该列表同原始数据混合
        #expand_list:[(img_path1,target1),(img_path2,target2)...]
        #temporarily表示暂时性的扩展，再次调用expand会替换掉之前扩展数据；设置为Fasle表明永久扩展该数据集,再次调用expand会保留之前传进去的扩展数据
        if temporarily:
            self.expand_samples = self.samples+expand_list
        else:
            self.samples.extend(expand_list)
            self.expand_samples = self.samples
        

class DataTxt(Dataset):
    #making train dataset from .txt file
    #the .txt file like this:
    #    image_path1 target1
    #    image_path2 target2  ...
    def __init__(self, txt_path, transform=None):
        self.txt_path = txt_path
        self.img_list = self._read_txt(txt_path)
        self.transform = transform
        self.loader = self._pil_loader
       
    def _read_txt(self, txt_path):
        with open(txt_path, 'r') as f:
            img_list = f.read().splitlines()
            return img_list
    
    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def __getitem__(self, index):
        img_path, target = self.img_list[index].split(' ')
        sample = self.loader(img_path)
        imag_name = os.path.basename(img_path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target, img_name
    
    def __len__(self):
        return len(self.img_list)
        
    
class TestDataset(Dataset):
    #making test dataset, transform is a function for transfering data
    def __init__(self,
                 root_dir,
                 transform=None,
                 pre_load=False):
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = os.listdir(self.root_dir)  #store filename
        self.images = None  #use later
        self.image_names = None
        self.loader = self._pil_loader
        self.len = len(self.filenames)
        if pre_load:
            self._pre_load()
        
    def _pre_load(self):  #preload the data
        self.images = []
        self.image_names = []
        
        for file in self.filenames:
            img = Image.open(os.path.join(self.root_dir, file))
            self.images.append(img.copy())
            img.close()
            self.image_names.append(file)
            
    def _pil_loader(self, path):
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
        
    def __getitem__(self,index):
        if self.images is not None:        #if pre_load
            img = self.images[index]
            img_name = self.image_name[index]
        else:
            img_name = self.filenames[index]
            img_path = os.path.join(self.root_dir, img_name)
            img = self.loader(img_path)
    
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name
        
    def __len__(self):
        return self.len



class data_prefetcher(object):
    #an interator
    #preload the data to GPU durning training
    def __init__(self, DataLoader):
        self.data_loader = iter(DataLoader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485*255, 0.456*255, 0.406*255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229*255, 0.224*255, 0.225*255]).cuda().view(1,3,1,1)
        self.preload()
        self._len = len(DataLoader)

    def preload(self):
        try:
            self.next_input, self.next_target = next(self.data_loader)
            #print(self.next_input.size(), self.next_target.size(), len(self.next_imgname))
        except StopIteration:
            self.next_input, self.next_target = None, None, None
            return
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def __iter__(self):
        return self
    
    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_input is not None:
            input = self.next_input
            target = self.next_target
            self.preload()
            return (input, target)
        else:
            raise StopIteration
    
    def __len__(self):
        return self._len
    
    
def fast_collate(batch):
    imgs = [img[0] for img in batch]
    print(imgs)
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w, h = imgs[0].size[0], imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        tens = torch.from_numpy(nump_array)
        if(nump_array.ndim <3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets
   

def collate_with_imgname_bak(batch):
    #imgs are PIL Image
    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    img_names = [img_name[2] for img_name in batch]
    w, h = imgs[0].size[0], imgs[0].size[1]
    tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.uint8)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim <3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets, img_names