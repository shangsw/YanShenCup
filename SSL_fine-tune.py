#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
半监督方法，使用测试集对模型进行微调
"""

from __future__ import print_function
import sys
import json
import argparse
import os
import shutil
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
# from flops_counter import get_model_complexity_info

from DataFunction import TestDataset, collate_with_imgname_bak, data_prefetcher, DataFolder
from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p
cudnn.benchmark = True
class_to_idx = {'停机坪': 0, '停车场': 1, '公园': 2, '公路': 3, '冰岛': 4, '商业区': 5, '墓地': 6, '太阳能发电厂': 7, '居民区': 8, '山地': 9, '岛屿': 10, '工厂': 11, '教堂': 12, '旱地': 13, '机场跑道': 14, '林地': 15, '桥梁': 16, '梯田': 17, '棒球场': 18, '水田': 19, '沙漠': 20, '河流': 21, '油田': 22, '油罐区': 23, '海滩': 24, '温室': 25, '港口': 26, '游泳池': 27, '湖泊': 28, '火车站': 29, '直升机场': 30, '石质地': 31, '矿区': 32, '稀疏灌木地': 33, '立交桥': 34, '篮球场': 35, '网球场': 36, '草地': 37, '裸地': 38, '足球场': 39, '路边停车区': 40, '转盘': 41, '铁路': 42, '风力发电站': 43, '高尔夫球场': 44}
# Models
default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names
#PATH
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--model_path', '-m', type=str, metavar='PATH',
                    help='path to model for test')
parser.add_argument('--datadir', type=str, metavar='DIR', 
                    help='path to train dir')
parser.add_argument('--testdir', type=str, metavar='DIR',
                   help='path to test directory')
#fine-tune setting
parser.add_argument('-th', '--threshold', default=0.95, type=float, metavar='N',
                    help='threshold of the belif for chosing samples for ssl')
parser.add_argument('--train_batch', type=int, metavar='N', default=32,
                    help='train batch size')
parser.add_argument('--test_batch', type=int, metavar='N', default=32,
                    help='test batch size')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--ft_epochs', default=5, type=int, metavar='N',
                    help='number of total fine-tune epochs to run')
parser.add_argument('--tr_epochs', default=30, type=int, metavar='N',
                    help='number of epochs in each fine-tune to train')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--schedule', type=int, nargs='+', default=[10,20,25],
                        help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint_ft', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
#use crop
parser.add_argument('--use_crop', dest='use_crop', action='store_true',
                    help='use crop to get average result')
#model arch
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('--modelsize', '-ms', metavar='large', default='large', \
                    choices=['large', 'small'], \
                    help = 'model_size affects the data augmentation, please choose:' + \
                           ' large or small ')
#Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}
# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

def main():
    # create model
    print("=> creating model '{}'".format(args.arch))
    if 'inception' in args.arch:
        #inceptionv3中的aux_logits是中间层的辅助输出,默认为True,model(inputs)将返回两个值
        model = model = models.__dict__[args.arch](aux_logits=False)  
    else:
        model = models.__dict__[args.arch]()

    if 'densenet' in args.arch:#这是因为densenet系列全连接层设置不一样
        in_features = model.classifier.in_features
        print(in_features)
        # sys.exit(1)
    else:
        in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 45)
    model = torch.nn.DataParallel(model).cuda()
    # print(model)

    #load weights
    print("=> loading weights '{}'".format(args.model_path))
    assert os.path.isfile(args.model_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    '''
    print("=> loaded checkpoint '{}' (epoch {})"
                            .format(args.model_path, checkpoint['epoch']))
    '''
    #data loading
    #print('=> Loading data from:', args.data_path)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
    data_aug_scale = (0.08, 1.0) if args.modelsize == 'large' else (0.2, 1.0)
    #训练集构建
    traindir = os.path.join(args.datadir, 'train')
    evaldir = os.path.join(args.datadir, 'eval')
    
    ssl_dataset = DataFolder(traindir, class_to_idx, transform=transforms.Compose([
        transforms.Resize((300,300)),
        transforms.RandomCrop(224),
        ]),pre_load=False)
    #验证集加入
    val_dataset = DataFolder(evaldir, class_to_idx, transform=transforms.Compose([
        transforms.Resize((224,224)),
        #transforms.RandomCrop(299),
        ]),pre_load=False)
    ssl_dataset.expand(val_dataset.samples)

    ssl_loader = torch.utils.data.DataLoader(
            ssl_dataset,
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True, collate_fn=collate_with_imgname_bak)
    #测试集构建
    if args.use_crop:
        test_dataset = TestDataset(args.testdir, transform=transforms.Compose([
                    transforms.Resize(300),
                    transforms.FiveCrop(224),
                    transforms.Lambda(lambda crops: torch.stack([normalize(transforms.ToTensor()(crop)) for crop in crops])), # returns a 4D tensor
                ]))
        print('Using crop.')
    else:
        test_dataset = TestDataset(args.data_path, transform=transforms.Compose([
                    transforms.Resize((300,300)),
                    #transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                    ]))

    test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.test_batch, shuffle=False,
            num_workers=args.workers, pin_memory=True)
    
    #fine-tune training settings
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)
    
    #暂时使用迭代次数作为终止微调的条件
    for ft_epoch in range(args.ft_epochs):
        print('-'*15,'Starting %d fine-tune' % ft_epoch,'-'*15)
        #获取伪标签集并更新训练集
        pseudo_samples = Get_pseudo_samples(model, test_loader, args.testdir, args.threshold)
        ssl_dataset.expand(pseudo_samples, temporarily=True)
        
        ssl_loader= torch.utils.data.DataLoader(
            ssl_dataset,
            batch_size=args.train_batch, shuffle=True,
            num_workers=args.workers, pin_memory=True, collate_fn=collate_with_imgname_bak)
        #微调网络
        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)  #训练参数设置
        fine_tune(model, ssl_loader, args.tr_epochs, criterion, optimizer, ft_epoch)
        #保存该轮fine-tune模型
        # save model
        model_path = os.path.join(args.checkpoint, str(ft_epoch)+'.pth.tar')
        torch.save({
                'ft_epochs': ft_epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, model_path)
        
   
    
def Get_pseudo_samples(model, test_loader, test_dir, th):
    '''
    对测试集进行前向推理，获得高置信度样本
    :model: 推理模型
    :test_loader: 测试集loader
    :th: 置信度阈值
    
    return：pseudo_samples:[(img_path1, pseudo_target1),(img_path2, pseudo_target2),...]
    '''
    print('-'*10+'Geting pseudo_samples...')
    print('=> Inferencing test data...')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    pd_imgnames_list = []
    #switch to evaluate mode
    model.eval()
    torch.set_grad_enabled(False)

    end = time.time()
    bar = Bar('Processing', max=len(test_loader))
    for batch_idx, (images, img_names) in enumerate(test_loader):
        #measure data loading time
        data_time.update(time.time()-end)
        #if use cuda:
        images = images.cuda()

        if args.use_crop:
            #images:[batch, ncrops, c, h, w]
            bs, ncrops, c, h, w = images.size()
            result = model(images.view(-1, c, h, w)) # fuse batch size and ncrops
            outputs = result.view(bs, ncrops, -1).mean(1) # avg over crops
        else:
            outputs = model(images)
        #对输出进行softmax获取置信度，并选取高置信度样本
        belifs = nn.Softmax()(outputs)  
        belif, pred = torch.max(belifs, 1)
        mask = belif.ge(th)
        mask_index = torch.nonzero(mask)
        pseudo_imgnames, pseudo_targets = [img_names[i] for i in mask_index], torch.masked_select(pred, mask)
        #print(mask, mask_index, pseudo_imgnames, pseudo_targets)
        #图像伪标签还是张量的形式
        if batch_idx == 0:
            pd_targets_list = pseudo_targets.clone()
        else:
            pd_targets_list = torch.cat((pd_targets_list, pseudo_targets))
        #print(pd_targets_list.size())
        pd_imgnames_list.extend(pseudo_imgnames)
        #print(len(pd_imgnames_list))

        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} '.format(
                        batch=batch_idx + 1,
                        size=len(test_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        )
        bar.next()
    print(bar.suffix)
    bar.finish()
    #pseudo samples
    pd_targets_list = pd_targets_list.cpu().numpy()
    pd_path_list = [os.path.join(test_dir, x) for x in pd_imgnames_list]
    pseudo_samples = list(zip(pd_path_list, pd_targets_list))
    print('Find %d pseudo samples.' % len(pseudo_samples))
    return pseudo_samples

def fine_tune(model, ps_loader, train_epochs, criterion, optimizer, ft_epoch):
    #使用伪标签样本对模型进行微调
    print('-'*10,'Starting fine-tuning...')
    #这个logger用来保存一轮fine-tune不同epoch的结果
    logger = Logger(os.path.join(args.checkpoint, str(ft_epoch)+'_log.txt'), title=str(args.arch+str(ft_epoch)+'ft'))  
    logger.set_names(['Learning Rate', 'Train Loss', 'Train Acc.'])
    
    model.train()
    
    for tr_epoch in range(train_epochs):
        adjust_learning_rate(optimizer, tr_epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (tr_epoch + 1, train_epochs, state['lr']))

        train_loss, train_acc = train(ps_loader, model, criterion, optimizer, tr_epoch, use_cuda)
        
        # append logger file
        logger.append([state['lr'], train_loss, train_acc])
        #保存此epoch结果
        model_path = os.path.join(args.checkpoint, 'checkpoint.pth.tar')
        torch.save({
                'train_epochs': tr_epoch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, model_path)

    logger.close()
    
def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    train_loader = data_prefetcher(train_loader)
    for batch_idx, (inputs, targets,_) in enumerate(train_loader):
        batch_size = inputs.size(0)
        
        if batch_size < args.train_batch:
            continue
        # measure data loading time
        data_time.update(time.time() - end)
        '''
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        '''
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        # top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                    batch=batch_idx + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg,
                    top1=top1.avg,
                    # top5=top5.avg,
                    )
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg)

def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

if __name__ == '__main__':
    main()