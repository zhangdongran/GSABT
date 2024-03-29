import os
import sys
import time
from datetime import datetime

import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random

# import metrics
import utils
import MinMax_utils
from utils import log_string, plot_train_val_loss

now_time = datetime.now()

# 一些训练参数，设置默认值，也可以在命令行指定值
parser = argparse.ArgumentParser("Traffic  prediction")
parser.add_argument('--model_file', default='./parameter/GSABT.pkl', help='save the model to disk')

parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='init learning rate')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--patience', type=int, default=30, help='patience')

parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--train_rate', type=float, default=48*28, help='train_rate')
parser.add_argument('--val_rate', type=float, default=48*14, help='val_rate')

parser.add_argument('--input_dim', type=int, default=12, help='input_dim')
parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim')
parser.add_argument('--output_dim', type=int, default=12, help='output_dim')
parser.add_argument('--blocks', type=int, default=2, help='blocks')
parser.add_argument('--feature', type=int, default=2, help='feature')
parser.add_argument('--dropout', type=int, default=0.1, help='dropout')
args = parser.parse_args()

def main():

    data_type = 'nyc'

    log_file='parameter/MPBTCN_{}_{: 04d}-{: 02d}-{: 02d}-{: 02d}-{: 02d}.txt'.\
        format(data_type, now_time.year, now_time.month, now_time.day, now_time.hour, now_time.minute)

    log = open(log_file, 'w')
    log_string(log, str(args)[10: -1])

    # load data
    log_string(log, 'loading data...')
    log_string(log, 'Blocks:{}, Seed:{}' .format(args.blocks, args.seed))

    train_loader, valid_loader, test_loader, mean, std, nodes, matrix, log = utils.s_taxi(args, log)


    log_string(log, 'nodes: {}'.format(nodes))
    log_string(log, 'data loaded!')

    # build model
    log_string(log, 'compiling model...')

    matrix = torch.from_numpy(matrix).float().to(device)

    from model.S_MPBTCN import Network

    model = Network(matrix, nodes, args.blocks, args.feature, args.input_dim, args.hidden_dim, args.output_dim, args.dropout).to(device)
    parameters=utils.count_parameters(model)
    log_string(log, 'trainable parameters: {:.2f}MB'.format(parameters))

    # L1损失函数
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

    best_loss = float('inf')
    best_epoch = 1
    wait = 0
    for epoch in range(1, args.epochs + 1):


        start = time.time()
        train_loss = train(train_loader, model, criterion, optimizer)
        valid_loss = valid(valid_loader, model, criterion)
        end = time.time()

        # 保存一下模型
        if valid_loss < best_loss:
            wait = 0
            best_epoch = epoch
            best_loss = valid_loss
            torch.save(model, args.model_file)

        else:
            wait = wait +1

        log_string(log, 'Epoch:{}, train_loss:{:.5f}, valid_loss:{:.5f},本轮耗时：{:.2f}s, best_epoch:{}, best_loss:{:.5f}'
                        .format(epoch, train_loss, valid_loss, end - start, best_epoch, best_loss))


    output, target = test(test_loader)

    output = output * std + mean
    target = target * std + mean

    log_string(log, '误差累计')

    error(output, target, args, log)

def error(output, target, args, log):

    Horizion = args.output_dim  # T = 12
    MAE = []
    RMSE = []
    PCC = []
    for i in range(Horizion):
        out = output[:, i, :, :]
        tgt = target[:, i, :, :]
        mae, rmse, pcc = utils.evalution(out, tgt)
        log_string(log, '第{}步的预测结果: MAE:{:.4f}, RMSE:{:.4f}, PCC:{:.4f}'.format(i + 1, mae, rmse, pcc))
        MAE.append(mae)
        RMSE.append(rmse)
        PCC.append(pcc)
    MAE = np.array(MAE).mean()
    RMSE = np.array(RMSE).mean()
    PCC = np.array(PCC).mean()
    log_string(log, 'MAE:{:.6f}, RMSE:{:.6f}, PCC:{:.6f}'.format(MAE, RMSE, PCC))



def train(train_loader, model, criterion, optimizer):
    # 记录训练误差
    train_loss = utils.AvgrageMeter()
    model.train()
    for step, (input, target) in enumerate(train_loader):
        # 当前的batch size，每个epoch的最后一个iteration的batch size不一定是设置的数值
        n = input.size(0)

        optimizer.zero_grad()
        input = Variable(input).to(device)
        target = Variable(target).to(device)

        output= model(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.data, n)
    return train_loss.avg

def valid(valid_loader, model, criterion):
    # 记录验证误差
    valid_loss = utils.AvgrageMeter()
    model.eval()
    with torch.no_grad():
        for step, (input, target) in enumerate(valid_loader):
            # 当前的batch size，每个epoch的最后一个iteration的batch size不一定是设置的数值
            n = input.size(0)

            input = Variable(input).to(device)
            target = Variable(target).to(device)

            output = model(input)
            loss = criterion(output, target)
            valid_loss.update(loss.data, n)

    return valid_loss.avg

def test(test_loader):
    torch.cuda.empty_cache()
    model = torch.load(args.model_file)
    model.eval()
    out = []
    tgt = []
    with torch.no_grad():
        for step, (input, target) in enumerate(test_loader):
            input = Variable(input).cuda()
            target = Variable(target).cuda()

            output = model(input)
            out.append(output)
            tgt.append(target)

    output = torch.cat(out, dim=0).cpu().numpy()
    target = torch.cat(tgt, dim=0).cpu().numpy()
    return output, target


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(args.seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(args.seed)  # Numpy module.
    random.seed(args.seed)  # Python random module.

    main()

    print('Model Finish!')


