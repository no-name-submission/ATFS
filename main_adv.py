from __future__ import print_function

import os
import sys
import argparse

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from util import *
from networks.resnet_big import SupCEResNet, SupConResNet
from losses import GraphLoss_1
from utils_awp import *
def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--epochs', type=int, default = 120,
                        help='number of training epochs')
    parser.add_argument('--adv_connect', type=bool, default=False,
                        help='how many edge adv sample connect to')

    parser.add_argument('--adv_upweight', type=int, default=0,
                        help='upweight the adv link')

    parser.add_argument('--init_epoch', type=int, default=10,
                        help='how many epoch to run before using graphloss')
    parser.add_argument('--init_path', default= './adv_init.pth',
                        help='init model')
    parser.add_argument('--save_per_epoch', type=int, default=40,
                        help='save_per_epoch')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning_rate')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='num classes')


    # model dataset
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--dataset', type=str, default='cifar10')

    # temperature
    parser.add_argument('--temp', type=float, default=1,
                        help='temperature for loss function')

    # other setting
    parser.add_argument('--load_from_ckpt', action='store_true',
                        help='load from checkpoint')
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--weight', type=float, default=1,
                        help='weight')
    parser.add_argument('--weight1', type=float, default=0.01, help = 'weight1')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    opt.model_name = 'GraphCE_{}_model_{}_bs_{}_temp_{}_alpha_{}_beta_{}'.\
        format(opt.dataset, opt.model, opt.batch_size, opt.temp, opt.weight, opt.weight1)

    #if opt.dataset == 'cifar100':
    #    opt.model_name = 'cifar100_{}'.format(opt.model_name)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    if opt.adv_connect != True:
        opt.model_name = '{}_advweight_{}'.format(opt.model_name, opt.adv_upweight + 1)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)

    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_model(opt):
    model = SupCEResNet(name=opt.model, num_classes=opt.num_classes)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print('available device:', torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        cudnn.benchmark = True

    return model

def contrastive_loss(clean_features, adv_features, labels, temp, adv_connect, upweight):

    criterion = GraphLoss_1(temp, base_temperature = temp)

    features = torch.cat([clean_features, adv_features], dim=0)
    loss_1 = criterion(features, labels, adv_connect_all = adv_connect, adv_upweight = upweight)

    return loss_1


def contrastive_train(train_loader, model, optimizer, epoch, opt):

    """one epoch training"""
    print('now epoch ' + str(epoch) + ': Doing Contrastive Adversarial Training', flush = True)

    configs1 =  {
    'epsilon': 8/255,
    'num_steps': 10,
    'step_size': 2/255,
    'clip_max': 1,
    'clip_min': 0
    }

    model.train()
    loss_1_sum = 0
    loss_2_sum = 0
    loss_sum = 0
    correct = 0

    for idx, (images, labels) in enumerate(train_loader):

        images, labels = torch.tensor(images).to('cuda'), torch.tensor(labels).to('cuda')

        clean_samples = torch.clone(images)
        if ( opt.weight != 0):
            adv_samples = pgd_attack(model, images, labels, **configs1)

        # compute adv loss
        logits_clean, features_clean = model(clean_samples, feat = True)
        #loss_clean = F.cross_entropy(logits_clean, labels)
        pred = logits_clean.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = correct + pred.eq(labels.view_as(pred)).sum().item()

        # compute adv loss
        if ( opt.weight != 0):
            logits_adv, features_adv = model(adv_samples, feat = True)
        #loss_adv = F.cross_entropy(logits_adv, labels)

        # contrastive loss
        if ( opt.weight != 0 ):
            loss_1 = contrastive_loss(features_clean, features_adv, labels, opt.temp, opt.adv_connect, opt.adv_upweight)
        else:
            loss_1 = torch.tensor(0).to('cuda')

        # clean loss
        loss_2 = F.cross_entropy(logits_clean, labels)

        # final loss
        loss = opt.weight * loss_1 + opt.weight1 * loss_2

        loss_1_sum = loss_1_sum + loss_1
        loss_2_sum = loss_2_sum + loss_2

        loss_sum = loss_sum + loss
        """
        if idx % 50 == 0:
            print('epoch:{:1f},\t idx:{:1f},\t loss1:{:.3f},\t loss2:{:.3f}'.format(epoch, idx, loss_1.item(), loss_2.item()))
        """
        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('epoch:{:1f},\t loss1:{:.3f},\t loss2:{:.3f},\t loss:{:.3f}\t, acc:{:.3f}'.format(epoch, loss_1_sum.item() / len(train_loader), loss_2_sum.item() / len(train_loader), loss_sum.item() / len(train_loader), correct / (len(train_loader) * opt.batch_size )))


def main(opt):

    # build model and criterion
    model = set_model(opt)

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build optimizer
    optimizer = optim.SGD(model.parameters(), lr = opt.learning_rate, momentum = 0.9, weight_decay = 5e-4)

    if (opt.dataset == 'cifar10'):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100], gamma = 0.1)
    elif (opt.dataset == 'cifar100'):
         lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75, 100, 150], gamma = 0.1)
    elif (opt.dataset == 'svhn'):
         lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma = 0.1)

    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30 ,75, 60, 90, 120, 150], gamma=0.1)
    #lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 150], gamma = 0.1)

    best_acc = 0
    # training routine
    for epoch in range(1, opt.epochs + 1):
        # train for one epoch

        if epoch <= opt.init_epoch:
            if (opt.load_from_ckpt):
                model.load_state_dict(torch.load(opt.init_path)['model'])
            else:
                pgd_train(train_loader, model, optimizer, epoch)

        else:
            contrastive_train(train_loader, model, optimizer, epoch, opt)
            ## check the test loss
            if epoch % 1 == 0:
                val_acc, adv_val_acc = validate(model, val_loader, 'cuda')

                if adv_val_acc > best_acc:
                    best_acc = adv_val_acc
                    save_file = os.path.join(
                            opt.save_folder, 'best.pth')
                    save_model(model, optimizer, opt, opt.epochs, save_file)

            if epoch % opt.save_per_epoch == 0:
                ## save model parameter
                save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
                save_model(model, optimizer, opt, epoch, save_file)

        lr_scheduler.step(epoch)

    # save the last model
    save_file = os.path.join(
                        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    print('best accuracy: {:.2f}'.format(best_acc))

if __name__ == '__main__':
    opt = parse_option()
    print(opt)
    main(opt)
