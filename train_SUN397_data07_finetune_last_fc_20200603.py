 # encoding:utf-8
'''
transfer learning : frozen some layers and only finetune the rest layers
===========================================================================
# CLQ-20200603   training SUN397     data07
nohup python -W ignore train_SUN397_data07_finetune_last_fc_20200603.py -a sksa_resnet101 --data /mnt/disk/home1/clq/datasets/SUN397/data07/ \
--epochs 100 --schedule 30 60 90 95  --lr 0.1 --train-batch 256 --test-batch 256 \
--gamma 0.1 -c checkpoints/SUN397/data07_sksa_resnet101_finetune_last_fc --gpu-id 1 \
>train_log_SUN397_data07_20200603.txt 2>&1 &



===========================================================================
# CLQ-20191224
(1) only finetune the last fc layer     flag_finetune_style = 0
-a sk_resnet101 --data /media/clq/Work/datasets/MIT67 --epochs 100 --schedule 30 60 90 95  --lr 0.01 --train-batch 32 --test-batch 24
--gamma 0.1 -c checkpoints/MIT67/sk_resnet101_frozen2_pretrained_Places365 --gpu-id 0

Best acc:85.2985076904   (epoch = 36)

********** change --lr 0.01 to --lr 0.0075
-a sk_resnet101 --data /media/clq/Work/datasets/MIT67 --epochs 100 --schedule 30 60 90 95  --lr 0.0075 --train-batch 32 --test-batch 24
--gamma 0.1 -c checkpoints/MIT67/sk_resnet101_frozen2_pretrained_Places365_lr0.0075 --gpu-id 0

Best acc:85.2238845825    (epoch = 19)
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
(2) finetune the last conv-stage and the last fc layer         flag_finetune_style = 1
-a sk_resnet101 --data /media/clq/Work/datasets/MIT67 --epochs 100 --schedule 30 60 90 95  --lr 0.01 --train-batch 32 --test-batch 24
--gamma 0.1 -c checkpoints/MIT67/sk_resnet101_frozen3_pretrained_Places365 --gpu-id 0

Best acc:83.4328384399   (epoch = 80)
===========================================================================
'''
from __future__ import print_function
import sys
import argparse
import os
import shutil
import time
import random

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import models.imagenet as customized_models
from flops_counter import get_model_complexity_info
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from utils import Bar, Logger, AverageMeter, accuracy, mkdir_p


# for servers to immediately record the logs
def flush_print(func):
    def new_print(*args, **kwargs):
        func(*args, **kwargs)
        sys.stdout.flush()

    return new_print

print = flush_print(print)

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

# Parse arguments
parser = argparse.ArgumentParser(description='PyTorch Training for Scene Recognition')

# Datasets
parser.add_argument('-d', '--data', default='/mnt/disk/home1/clq/datasets/MIT67', type=str)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
# Optimization options
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--train-batch', default=32, type=int, metavar='N',
                    help='train batchsize (default: 256)')
parser.add_argument('--test-batch', default=16, type=int, metavar='N',
                    help='test batchsize (default: 200)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--drop', '--dropout', default=0, type=float,
                    metavar='Dropout', help='Dropout ratio')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
# Checkpoints
parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH',
                    help='path to save checkpoint (default: checkpoint)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# Architecture
parser.add_argument('--modelsize', '-ms', metavar='large', default='large', \
                    choices=['large', 'small'], \
                    help='model_size affects the data augmentation, please choose:' + \
                         ' large or small ')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
# added by CLQ
parser.add_argument('--num_classes', default=397, type=int, help='num of class in the model')
parser.add_argument('--dataset', default='SUN397', help='which dataset to train')

parser.add_argument('--depth', type=int, default=29, help='Model depth.')
parser.add_argument('--cardinality', type=int, default=32, help='ResNet cardinality (group).')
parser.add_argument('--base-width', type=int, default=4, help='ResNet base width.')
parser.add_argument('--widen-factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
# Miscs
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
# Device options
parser.add_argument('--gpu-id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

args = parser.parse_args()
state = {k: v for k, v in args._get_kwargs()}

# Use CUDA
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
use_cuda = torch.cuda.is_available()

# Random seed
if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)

best_acc = 0  # best test accuracy


def main():
    global best_acc
    print(args)
    start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'test')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    data_aug_scale = (0.08, 1.0) if args.modelsize == 'large' else (0.2, 1.0)

    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224, scale=data_aug_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.test_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # create model
    print(args.pretrained)
    Pretrained_ImageNet_or_Places365 = 0    # 1: ImageNet     0: Places365
    if args.arch.startswith('densenet'):
        model = load_pretrained_Places365_model(args, '/media/clq/Work/datasets/pretrained_models/densenet161_places365.pth.tar')
    elif args.arch=='sk_resnet101' or args.arch=='sksa_resnet101':
        if Pretrained_ImageNet_or_Places365 ==1:
            print('Training {} with pretrained ImageNet models on MIT67'.format(args.arch))
            model = load_pretrained_ImageNet_model(args,
                                                   '/media/clq/Work/datasets/pretrained_models/sk_resnet101.pth.tar')
        else:
            print('clq 0601 Training {} with pretrained Places365 models on MIT67'.format(args.arch))
            model = load_pretrained_ImageNet_model(args,
                                                   '/mnt/disk/home1/clq/PytorchInsight/classification/checkpoints/Places365_standard/sksa_resnet101_2/model_best.pth.tar', Pretrained_ImageNet_or_Places365 = 0)
    elif  args.arch=='sk_resnet50':
        model = load_pretrained_ImageNet_model(args, '/media/clq/Work/datasets/pretrained_models/sk_resnet150.pth.tar')

    # Frozen some layers
    # 在训练时如果想要固定网络的底层，那么可以令这部分网络对应子图的参数requires_grad为False。
    # 这样，在反向过程中就不会计算这些参数对应的梯度
    for param in model.parameters():
        param.requires_grad = False

    flag_finetune_style = 0      # 1: finetune stage 4 and last fc     0: only finetune last fc
    if flag_finetune_style==1:
        for param in model.layer4.parameters():
            param.requires_grad = True      # finetune the last conv stage
    else:
        pass       # only finetune the last fc layer
        print('only finetune the last fc layer')

    # 修改类别数
    if args.arch.startswith('resnet') or args.arch.startswith('sk_resnet') or args.arch.startswith('sksa_resnet'):
        in_ftrs = model.fc.in_features  # 最后一层全连接层
        print('pretrained  model.fc.size={}*{}'.format(model.fc.in_features, model.fc.out_features))
        model.fc = nn.Linear(in_ftrs, args.num_classes)  # 修改类别数
    if args.arch.startswith('densenet'):
        in_ftrs = model.classifier.in_features  # 最后一层全连接层
        print('pretrained  model.fc.size={}*{}'.format(model.classifier.in_features, model.classifier.out_features))
        model.fc = nn.Linear(in_ftrs, args.num_classes)  # 修改类别数

    # Optimize only the classifier
    if flag_finetune_style==1:
        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, model.parameters()),      # 记住一定要加上filter()，不然会报错
            lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    print(model)
    flops, params = get_model_complexity_info(model, (224, 224), as_strings=False, print_per_layer_stat=False)
    print('Flops:  %.3f' % (flops / 1e9))
    print('Params: %.2fM' % (params / 1e6))

    # 多GPU训练
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model).cuda()

    cudnn.benchmark = True

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)


    # Resume
    title = 'MIT67-' + args.arch
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..', args.resume)
        assert os.path.isfile(args.resume), 'Error: no checkpoint directory found!'
        args.checkpoint = os.path.dirname(args.resume)
        checkpoint = torch.load(args.resume)
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        # model may have more keys
        t = model.state_dict()
        c = checkpoint['state_dict']
        #flag = True
        # for k in t:
        #     if k not in c:
        #         print('not in loading dict! fill it', k, t[k])
        #         c[k] = t[k]
        #         flag = False
        # model.load_state_dict(c)
        flag = False      # modified by CLQ
        for k in c:
            if k.startswith('module'):
                t[k[7:]]=c[k]
        model.load_state_dict(t)
        if flag:
            print('optimizer load old state')
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('new optimizer !')
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title, resume=True)
    else:
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title=title)
        logger.set_names(
            ['epoch', 'LR', 'Train Loss', 'Valid Loss', 'Train Top1', 'Valid Top1.', 'Train Top5', 'Valid Top5'])

    #args.evaluate=True
    print('args.evaluate:{}'.format(args.evaluate))
    if args.evaluate:
        print('\nEvaluation only')
        test_loss, test_acc, test_top5  = test(val_loader, model, criterion, start_epoch, use_cuda)
        print(' Test Loss:  %.8f, Test Acc:  %.2f, Test Top5: %.2f' % (test_loss, test_acc, test_top5))
        return

    # Train and val
    for epoch in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        print('\nEpoch: [%d | %d] LR: %f' % (epoch + 1, args.epochs, state['lr']))

        train_loss, train_acc, train_top5 = train(train_loader, model, criterion, optimizer, epoch, use_cuda)
        test_loss, test_acc, test_top5 = test(val_loader, model, criterion, epoch, use_cuda)

        # append logger file
        logger.append([int(epoch), state['lr'], train_loss, test_loss, train_acc, test_acc, train_top5, test_top5])

        # save model
        is_best = test_acc > best_acc
        best_acc = max(test_acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'acc': test_acc,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, checkpoint=args.checkpoint)

    print('Best acc:{}'.format(best_acc))
    print(args)
    logger.set_names(['Best acc'])
    logger.append([best_acc])
    logger.close()


def train(train_loader, model, criterion, optimizer, epoch, use_cuda):
    # switch to train mode
    model.train()
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    show_step = len(train_loader) // 10
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        batch_size = inputs.size(0)
        if batch_size < args.train_batch:
            continue
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda(async=True)
        inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        losses.update(loss.data, inputs.size(0))
        top1.update(prec1, inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        if (batch_idx) % show_step == 0:
            print(bar.suffix)
        bar.next()
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)


def test(val_loader, model, criterion, epoch, use_cuda):
    global best_acc

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    torch.set_grad_enabled(False)

    end = time.time()
    bar = Bar('Processing', max=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = torch.autograd.Variable(inputs, volatile=True), torch.autograd.Variable(targets)

        # compute output
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(outputs.data, targets.data, topk=(1, 5))
        # losses.update(loss.data[0], inputs.size(0))
        losses.update(loss.data, inputs.size(0))
        # top1.update(prec1[0], inputs.size(0))
        top1.update(prec1, inputs.size(0))
        # top5.update(prec5[0], inputs.size(0))
        top5.update(prec5, inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '(Test {batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=batch_idx + 1,
            size=len(val_loader),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    print(bar.suffix)
    bar.finish()
    return (losses.avg, top1.avg, top5.avg)


def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))


def adjust_learning_rate(optimizer, epoch):
    global state
    if epoch in args.schedule:
        state['lr'] *= args.gamma
        for param_group in optimizer.param_groups:
            param_group['lr'] = state['lr']

#====================================================================
# written by CLQ
def load_pretrained_ImageNet_model(args, model_file, Pretrained_ImageNet_or_Places365 = 1):
    '''' To load pretrained sk_resnet model '''
    if not os.access(model_file, os.W_OK):
        print('Please download pretrained models ! ')
    if Pretrained_ImageNet_or_Places365==1:
        model = models.__dict__[args.arch]()     # ImageNet model (default num_classes=1000)
    else:
        model = models.__dict__[args.arch](num_classes=365)     # Places365 model

    for name in model.state_dict():
       print(name)
    emptydict=model.state_dict()

    checkpoint = torch.load(model_file)
    pretraineddict=checkpoint['state_dict']
    i=0
    for name in checkpoint['state_dict']:
        i=i+1
        print('pretrained models  the {} layer   :  {}  '.format(i, name))
        if name.startswith('module'):
            name00=name[7:]
            print('our modified models the {} layer   :  {}  '.format(i, name00))
            emptydict[name00]=pretraineddict[name]     #initialization
    model.load_state_dict(emptydict)
    print(model)
    return model


def load_pretrained_Places365_model(args, model_file):
    '''' To load pretrained DenseNet model '''
    if not os.access(model_file, os.W_OK):
        print('Please download pretrained models ! ')
    model = models.__dict__[args.arch](num_classes=365)
    # for name in model.state_dict():
    #    print(name)
    emptydict=model.state_dict()

    checkpoint = torch.load(model_file)
    pretraineddict=checkpoint['state_dict']
    i=0
    j=0
    for name in checkpoint['state_dict']:
        i=i+1
        #print('pretrained models  the {} layer   :  {}  '.format(i, name))
        if name.startswith('module'):
            name00=name[7:]
            if 'transition' in name00:    #  module.features.transition1.norm.weight
                name00 = name00        #                 features.transition1.norm.weight
            else:
                if 'conv.' in name00:
                    name00 = name00.replace('conv.', 'conv')
                elif 'norm.' in name00:      # name = module.features.denseblock1.denselayer1.norm.1.weight
                    name00 = name00.replace('norm.', 'norm')      #  features.denseblock1.denselayer1.norm1.weight
            # ensure the layer in our model
            if name00 in emptydict:
                j=j+1
                print('our modified models the {} layer   :  {}  '.format(j, name00))
                emptydict[name00] = pretraineddict[name]  # initialization
            else:
                print(i)
                print('some layers not in pretrained models    {}  :   {}'.format(i, name00))
    model.load_state_dict(emptydict)
    print('Total layers in checkpoint : {}           number of the same layers:{}'.format(i, j))
    print(model)
    return model


if __name__ == '__main__':
    main()