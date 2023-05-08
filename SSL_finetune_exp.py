import math
import os
import random
import shutil
import warnings
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from q3_solution  import SimSiam
from q3_misc import TwoCropsTransform, load_checkpoints, load_pretrained_checkpoints
from utils.func_tool import save_logs

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# general
seed = 2022
num_workers = 2
save_path1 = Path("./SSL_FineTune_StopG_NoneMLP")
save_path1.mkdir(exist_ok = True)
save_path = "./SSL_FineTune_StopG_NoneMLP"
resume = None # None or a path to a pretrained model (e.g. *.pth.tar')
start_epoch = 0
epochs = 100 # Number of epoches (for this question 200 is enough, however for 1000 epoches, you will get closer results to the original paper)

# data
dir ='./data'
batch_size = 1024

# Siamese backbone model
arch = "resnet18"
fix_pred_lr = True # fix the learning rate of the predictor network

#Simsiam params
dim=2048 # 2048 | 4096
pred_dim=512

# ablation experiments
stop_gradient=True # (True or False)
MLP_mode=None # None|'no_pred_mlp'|'fixed_random_init'

# optimizer
lr = 0.03
momentum = 0.9
weight_decay = 0.0005

random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True
# Simsiam Model
print("=> creating model '{}'".format(arch))
model = SimSiam(models.__dict__[arch], dim, pred_dim, stop_gradient=stop_gradient, MLP_mode=MLP_mode)
model.to(device)

# define and set learning rates
init_lr = lr * batch_size / 256
if fix_pred_lr:
    optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                    {'params': model.predictor.parameters(), 'fix_lr': True}]
else:
    optim_params = model.parameters()

# define optimizer
optimizer = torch.optim.SGD(optim_params, init_lr,
                            momentum=momentum,
                            weight_decay=weight_decay)
# adjust LR
def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


# linear eval
print("=> creating model '{}'".format(arch))
model = models.__dict__[arch]()

# freeze all layers but the last fc
for name, param in model.named_parameters():
    if name not in ['fc.weight', 'fc.bias']:
        param.requires_grad = False

# init the fc layer
model.fc.weight.data.normal_(mean=0.0, std=0.01)
model.fc.bias.data.zero_()
print(model)

# load the pre-trained model from previous steps
pretrained = 'runs/stopGrad_NoneMLP/checkpoint_0099.pth.tar'
# pretrained = '../SSL_results/stopGrad_NoneMLP/checkpoint_0099.pth.tar'

if pretrained:
    model, optimizer, start_epoch = load_pretrained_checkpoints(os.path.join(pretrained),model,optimizer,device)
if device is not None:
    model.to(device)

# define loss function (criterion) and optimizer
criterion = nn.CrossEntropyLoss().to(device)

# optimize only the linear classifier
parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
assert len(parameters) == 2  # fc.weight, fc.bias

optimizer = torch.optim.SGD(parameters, init_lr,
                            momentum=momentum,
                            weight_decay=weight_decay)

# define train and test augmentation for linear evaluation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_dataset = datasets.CIFAR10(
    dir,
    transform=train_transform,
    download=True,
    train=True
    )
val_dataset = datasets.CIFAR10(
    dir,
    transform=val_transform,
    download=True,
    train=False
    )


train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, #(train_sampler is None),
    num_workers=num_workers, pin_memory=True) #, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=256, shuffle=False,
    num_workers=num_workers, pin_memory=True)



# train for one epoch
def train(train_loader, model, criterion, optimizer, device):

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    losses=[]
    top1=[]
    top5=[]
    for i, (images, target) in enumerate(train_loader):

        if device is not None:
            images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.append(loss.item())
        top1.append(float(acc1[0].cpu()))
        top5.append(float(acc5[0].cpu()))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return top1, top5

# validation
def validate(val_loader, model, criterion, device):

    # switch to evaluate mode
    model.eval()
    losses=[]
    top1=[]
    top5=[]
    with torch.no_grad():

        for i, (images, target) in enumerate(val_loader):
            if device is not None:
                images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.append(loss.item())
            top1.append(float(acc1[0].cpu()))
            top5.append(float(acc5[0].cpu()))

    return top1, top5


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# train for the classififcation task

logger_c = dict()
logger_c['train_top1'] = [0]
logger_c['val_top1'] = [0]
logger_c['train_top5'] = [0]
logger_c['val_top5'] = [0]
for epoch in range(start_epoch, epochs):

    adjust_learning_rate(optimizer, init_lr, epoch, epochs)

    # train for one epoch
    acc1_1, acc1_5 = train(train_loader, model, criterion, optimizer,device)
    print('Train Epoch: [{}/{}] Train acc1:{:.2f}%'.format(epoch, epochs,np.array(acc1_1).mean() ))
    logger_c['train_top1'].append(np.array(acc1_1).mean())
    logger_c['train_top5'].append(np.array(acc1_5).mean())

    # evaluate on validation set
    acc2_1, acc2_5 = validate(val_loader, model, criterion, device)
    print('Val Epoch: [{}/{}] Val acc1:{:.2f}%'.format(epoch, epochs,np.array(acc2_1).mean() ))
    logger_c['val_top1'].append(np.array(acc2_1).mean())
    logger_c['val_top5'].append(np.array(acc2_5).mean())


save_logs(logger_c, "SSL_FineTune_StopG_NoneMLP/log_new_c", str(1))













