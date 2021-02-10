from __future__ import print_function

import os
import sys
import argparse
import time
import math

import tensorboard_logger as tb_logger
import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
import torch.nn.functional as F
#from util import AverageMeter
#from util import adjust_learning_rate, warmup_learning_rate
from util import accuracy, validate, validate_autoattack
from util import set_optimizer, save_model
from networks.resnet_big import SupCEResNet
from networks.wide_resnet import SupCEWRN
from deeprobust.image.attack.pgd import PGD
from autoattack import AutoAttack

from resnet import *
from config import *
from losses import GraphLoss
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=80,
                        help='number of training epochs')
    parser.add_argument('--adv_connect', type=bool, default=True,
                        help='how many edge adv sample connect to')
    parser.add_argument('--init_epoch', type=int, default=30,
                        help='how many epoch to run before using graphloss')
    parser.add_argument('--temp', type=float, default=0.1,
                        help='temperature for loss function')
    parser.add_argument('--ep', type=float, default=8.0,
                        help='perturbation')
    parser.add_argument('--advweight', type=int, default = 1)


    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.2,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='70,120',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='dataset')
    parser.add_argument('--model_name', type=str, default='./')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--weight', type=float, default=0.1,
                        help='weight')


    opt = parser.parse_args()

    opt.data_folder = './datasets/'
    opt.model_path = './save2/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save2/SupCon/{}_tensorboard'.format(opt.dataset)

    #opt.model_name = 'GraphCE_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
    #    format(opt.dataset, opt.model, opt.learning_rate, opt.weight_decay,
    #           opt.batch_size, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)

    #opt.model_name = 'GraphCE_cifar10_model_resnet18_temp_0.1_alpha_100.0_beta_1.0_cosine'
    #opt.model_name = 'GraphCE_cifar10_model_resnet18_temp_0.1_alpha_1.0_beta_0.01_cosine'
    #opt.model_name = 'GraphCE_cifar100_model_resnet18_temp_0.1_alpha_1.0_beta_0.1_cosine'
    #opt.model_name = 'GraphCE_cifar10_model_resnet18_temp_0.1_alpha_1.0_beta_0.01_cosine_advweight_2'
    #opt.model_name = 'GraphCE_cifar10_model_resnet18_temp_0.1_alpha_0.0_beta_1.0_cosine_advweight_1'
    #opt.model_name = 'GraphCE_cifar10_model_resnet18_temp_0.1_alpha_0.0_beta_1.0_cosine_advweight_1'
    #opt.model_name = 'GraphCE_cifar10_model_resnet18_temp_0.1_alpha_1.0_beta_0.01_cosine_advweight_2'
    #opt.model_name = 'GraphCE_cifar10_model_WRN28_temp_0.1_alpha_1.0_beta_0.01_cosine_advweight_2'

    #opt.model_name = 'GraphCE_cifar10_model_resnet18_temp_0.1_alpha_1.0_beta_0.1_cosine_advweight_1'

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

def pgd_attack(model,
                  X,
                  y,
                  epsilon,
                  clip_max,
                  clip_min,
                  num_steps,
                  step_size):

    out = model(X)
    err = (out.data.max(1)[1] != y.data).float().sum()
    #TODO: find a other way
    device = X.device
    imageArray = X.detach().cpu().numpy()
    X_random = np.random.uniform(-epsilon, epsilon, X.shape)
    imageArray = np.clip(imageArray + X_random, 0, 1.0)

    X_pgd = torch.tensor(imageArray).to(device).float()
    X_pgd.requires_grad = True

    for i in range(num_steps):

        pred = model(X_pgd)
        loss = nn.CrossEntropyLoss()(pred, y)
        loss.backward()

        eta = step_size * X_pgd.grad.data.sign()

        X_pgd = X_pgd + eta
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)

        X_pgd = X.data + eta
        X_pgd = torch.clamp(X_pgd, clip_min, clip_max)
        X_pgd = X_pgd.detach()
        X_pgd.requires_grad_()
        X_pgd.retain_grad()

    return X_pgd


def set_loader(opt, model):
    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_adv_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])


    val_transform = transforms.Compose([
        transforms.ToTensor(),
        ])

    if opt.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=opt.data_folder,
                                         transform=train_transform,
                                         download=True)
        val_dataset = datasets.CIFAR10(root=opt.data_folder,
                                       train=False,
                                       transform=val_transform)

    elif opt.dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=opt.data_folder,
                                          transform=train_transform,
                                          download=True)
        val_dataset = datasets.CIFAR100(root=opt.data_folder,
                                        train=False,
                                        transform=val_transform)
    else:
        raise ValueError(opt.dataset)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=128, shuffle=False,
        num_workers=8, pin_memory=True)

    return train_loader, val_loader

def set_trade_model(opt):
    model = ResNet18().to('cuda')
    state_dict = torch.load('./model-resnet18-lambda5.0-best.pt')

    return model

def set_model(opt):
    if (opt.model == 'WRN28'):
        model = SupCEWRN(name=opt.model, num_classes=opt.n_cls)
    else:
        model = SupCEResNet(name=opt.model, num_classes=opt.n_cls)
    criterion = GraphLoss(opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        #if torch.cuda.device_count() > 1:
        #    print('available device:', torch.cuda.device_count())
        #    model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    state_dict = torch.load(opt.model_name + '/best.pth')['model']
    #state_dict = torch.load('./resnet18_clean_acc93.pth')['model']

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if (k[0:7] == 'module.'):
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    return model, criterion

def property(model, val_loader, configs):

    features = [[] for i in range(10)]
    features_adv = [[] for i in range(10)]
    for idx, (images, labels) in enumerate(val_loader):

        images, labels = images.to('cuda'), labels.to('cuda')
        score, feature = model(images, feat = True)

        norm1 = feature @ feature.T
        norm1 = torch.sqrt(torch.diag(norm1)).unsqueeze(1)
        feature = feature / norm1

        #clean_samples = torch.clone(images)
        adv_samples = pgd_attack(model, images, labels, **configs)
        score_adv, feature_adv = model(adv_samples, feat = True)


        norm2 = feature_adv @ feature_adv.T
        norm2 = torch.sqrt(torch.diag(norm2)).unsqueeze(1)
        feature_adv = feature_adv / norm2


        feature = feature.detach().to('cpu').numpy()
        feature_adv = feature_adv.detach().to('cpu').numpy()

        labels = labels.detach().to('cpu').numpy()
        for i in range(len(labels)):
            features[labels[i]].append(feature[i])
            features_adv[labels[i]].append(feature_adv[i])

        if (idx > 10):
            break

    for i in range(10):
        features[i] = np.array(features[i])
    mul = np.zeros((10, 10))

    for i in range(10):
        for j in range(10):
            mul[i][j] = (features[i] @ features[j].T).mean()

    return mul

def main():
    best_acc = 0
    opt = parse_option()

    # build model and criterion
    #model, criterion = set_trade_model(opt)
    #model = set_trade_model(opt)

    model, _ = set_model(opt)

    #model = ResNet18().to('cuda')
    #model = model.load_state_dict(torch.load('resnet18_clean_acc93.pth')['net'])
    """
    state_dict = torch.load('resnet18_clean_acc93.pth')['net']
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if (k[0:7] == 'module.'):
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        else:
            name = k
            new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    """

    # build data loader
    train_loader, val_loader = set_loader(opt, model)

    # build optimizer
    #optimizer = set_optimizer(opt, model)

    # tensorboard
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # set up attacker
    #attacker = PGD(model, 'cuda')
    #adversary = AutoAttack(model, norm='Linf', eps= 8/255, version='standard')

    # training routine
    # evaluation
    clean_accs = []
    adv_accs = []

    configs1 =  {
    'epsilon': 8/255,
    'num_steps': 20,
    'step_size': 2/255,
    'clip_max': 1,
    'clip_min': 0
    }

    import seaborn as sns
    sns.set_theme(style="darkgrid")


    """
    Visualization
    """
    """
    for idx, (images, labels) in enumerate(val_loader):

        images, labels = images.to('cuda'), labels.to('cuda')
        logits, feature = model(images, feat = True)

        #clean_samples = torch.clone(images)
        adv_samples = pgd_attack(model, images, labels, **configs1)

        logits_adv, feature_adv = model(adv_samples, feat = True)
        pred1 = feature_adv.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        add1 = pred1.eq(labels.view_as(pred1)).sum().item()
        print(add1/ len(images))

        #np.savetxt('adv_X.txt', feature_adv.detach().to('cpu'))
        #np.savetxt('labels.txt', labels.detach().to('cpu'))
        #np.savetxt('clean_X.txt', feature.detach().to('cpu'))
        feature = feature.detach().to('cpu').numpy()
        feature_adv = feature_adv.detach().to('cpu').numpy()

        labels = labels.detach().to('cpu').numpy()
        if (idx == 0):
            X = feature
            X_adv = feature_adv
            y = labels
        else:
            X = np.concatenate((X, feature))
            X_adv = np.concatenate((X_adv, feature))

            y = np.concatenate((y, labels))

        if (idx >= 5):
            break
    """

    """

    from sklearn import manifold,datasets
    import matplotlib.pyplot as plt

    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)


    X_tsne = tsne.fit_transform(X)


    X_adv_tsne = tsne.fit_transform(X_adv)

    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    x_adv_min, x_adv_max = X_adv_tsne.min(0), X_adv_tsne.max(0)
    X_adv_norm = (X_adv_tsne - x_adv_min) / (x_adv_max - x_adv_min)
    plt.figure(figsize=(8, 8))

    for i in range(X_norm.shape[0]):
        plt.plot(X_norm[i, 0], X_norm[i, 1], 'o', color = plt.cm.Set1(y[i]))

    for i in range(X_adv_norm.shape[0]):
        plt.text(X_adv_norm[i, 0], X_adv_norm[i, 1], 'o', color=plt.cm.Set1(y[i]))
        #plt.text(X_adv_norm[i, 0], X_adv_norm[i, 1], str(y[i]), color = plt.cm.Set1(y[i]), fontdict = {'fontsize': 9, 'fontweight': 'bold'})

    plt.title('ATFS (ep = 8.0/255)', fontsize = 20)
    labels = [0,1,2,3,4,5,6,7,8,9]
    patches = [ mpatches.Patch(color=plt.cm.Set1(labels[i]), label=str(labels[i]) ) for i in range(len(labels)) ]
    plt.legend(handles=patches, ncol=1, loc = 'right') #生成legend
    plt.xticks([])
    plt.yticks([])
    plt.show()
    plt.savefig('tsne_ATFS.png')
    """

    """
    heatmap
    """

    mean = property(model, val_loader, configs1)
    print(mean)

    hm = sns.heatmap(mean, cmap="YlGnBu")
    #plt.title('ATFS Feature correlation', fontsize = 22)
    plt.savefig('heatmap_ATFS.pdf')

if __name__ == '__main__':
    main()
