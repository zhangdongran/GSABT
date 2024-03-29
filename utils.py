import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader,TensorDataset
from sklearn.metrics import mean_squared_error,mean_absolute_error
import matplotlib.pyplot as plt


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


# plot train_val_loss
def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


# 用于计算平均值
class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def look_back(data, input_dim, output_dim):
    train = []
    test = []
    L = len(data)
    for i in range(L - input_dim - output_dim + 1):
        train_seq = data[i:i + input_dim, :, :]
        train_label = data[i + input_dim:i + input_dim + output_dim, :, :]
        train.append(train_seq)
        test.append(train_label)
    train = np.array(train)
    test = np.array(test)
    train = torch.FloatTensor(train)
    test = torch.FloatTensor(test)

    return train, test

def tbt(args, log):
    bj_in  = pd.read_csv(r'data\beijing\bj_in.csv', index_col=None, header=None).values  # L, N
    bj_out = pd.read_csv(r'data\beijing\bj_out.csv', index_col=None, header=None).values  # L, N

    taxi_pick = pd.read_csv(r'data\nyctaxi\taxi_pick.csv', index_col=None, header=None).values  # L, N
    taxi_drop = pd.read_csv(r'data\nyctaxi\taxi_drop.csv', index_col=None, header=None).values

    bike_pick = pd.read_csv(r'data\nycbike\bike_pick.csv', index_col=None, header=None).values
    bike_drop = pd.read_csv(r'data\nycbike\bike_drop.csv', index_col=None, header=None).values
    matrix = pd.read_csv(r'data\btb_matrix.csv', index_col=None, header=None).values
    f_nodes, s_nodes, t_nodes = len(bj_in[0]), len(taxi_pick[0]), len(bike_pick[0])
    nodes = f_nodes + s_nodes + t_nodes
    length = len(taxi_pick)

    bj = np.stack((bj_in, bj_out), axis=2)  # 4368, 256, 2
    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 266, 2
    bike = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 250, 2

    f_mean, f_std = bj.mean(), bj.std()
    bj = (bj - f_mean) / f_std

    s_mean, s_std = taxi.mean(), taxi.std()
    taxi= (taxi - s_mean) / s_std

    t_mean, t_std = bike.mean(), bike.std()
    bike = (bike - t_mean) / t_std

    print('bj.shape:', bj.shape)
    print('taxi.shape:', taxi.shape)
    print('bike.shape:', bike.shape)

    flow = np.concatenate((bj, taxi, bike), axis=1)  # 4368, 772, 2
    print('flow.shape:', flow.shape)

    train_rate, val_rate = int(args.train_rate), int(args.val_rate)
    train, valid, test = flow[0:-train_rate, :, :], flow[-train_rate:-val_rate, :, :], flow[-val_rate:, :, :]

    # 转变为 torch格式
    trainX, trainY = look_back(train, args.input_dim, args.output_dim)
    validX, validY = look_back(valid, args.input_dim, args.output_dim)
    testX, testY = look_back(test, args.input_dim, args.output_dim)

    log_string(log, f'trainX: {trainX.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validX.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:   {testY.shape}')

    train = TensorDataset(trainX, trainY)
    valid = TensorDataset(validX, validY)
    test = TensorDataset(testX, testY)
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, f_mean, f_std, s_mean, s_std, t_mean, t_std, nodes, f_nodes, s_nodes, t_nodes, matrix, log


def d_nyc(args, log):
    taxi_pick = pd.read_csv(r'data\nyctaxi\taxi_pick.csv', index_col=None, header=None).values  # L, N
    taxi_drop = pd.read_csv(r'data\nyctaxi\taxi_drop.csv', index_col=None, header=None).values
    bike_pick = pd.read_csv(r'data\nycbike\bike_pick.csv', index_col=None, header=None).values
    bike_drop = pd.read_csv(r'data\nycbike\bike_drop.csv', index_col=None, header=None).values
    matrix = pd.read_csv(r'data\nyc_matrix.csv', index_col=None, header=None).values

    f_nodes, s_nodes = len(taxi_pick[0]), len(bike_pick[0])
    nodes = f_nodes + s_nodes
    length = len(taxi_pick)

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 266, 2
    bike = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 250, 2

    f_mean, f_std = taxi.mean(), taxi.std()
    taxi= (taxi-f_mean)/f_std

    s_mean, s_std = bike.mean(), bike.std()
    bike = (bike - s_mean) / s_std

    print('taxi.shape:', taxi.shape)
    print('bike.shape:', bike.shape)

    flow = np.concatenate((taxi, bike), axis=1)  # 4368, 516, 2
    print('flow.shape:', flow.shape)

    train_rate, val_rate = int(args.train_rate), int(args.val_rate)
    train, valid, test = flow[0:-train_rate, :, :], flow[-train_rate:-val_rate, :, :], flow[-val_rate:, :, :]

    # 转变为 torch格式
    trainX, trainY = look_back(train, args.input_dim, args.output_dim)
    validX, validY = look_back(valid, args.input_dim, args.output_dim)
    testX, testY = look_back(test, args.input_dim, args.output_dim)

    log_string(log, f'trainX: {trainX.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validX.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:   {testY.shape}')

    train = TensorDataset(trainX, trainY)
    valid = TensorDataset(validX, validY)
    test = TensorDataset(testX, testY)
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_en = 0

    return train_loader, valid_loader, test_loader, f_mean, f_std, s_mean, s_std, nodes, f_nodes, s_nodes, matrix, log

def d_bjtaxi(args, log):
    bj_in = pd.read_csv(r'data\beijing\bj_in.csv', index_col=None, header=None).values  # L, N
    bj_out = pd.read_csv(r'data\beijing\bj_out.csv', index_col=None, header=None).values

    taxi_pick = pd.read_csv(r'data\nyctaxi\taxi_pick.csv', index_col=None, header=None).values  # L, N
    taxi_drop = pd.read_csv(r'data\nyctaxi\taxi_drop.csv', index_col=None, header=None).values
    matrix = pd.read_csv(r'data\bjtaxi.csv', index_col=None, header=None).values
    matrix = matrix.astype(float)

    f_nodes, s_nodes = len(bj_in[0]), len(taxi_pick[0])
    nodes = f_nodes + s_nodes

    len_taxi = len(taxi_pick)

    bj = np.stack((bj_in, bj_out), axis=2)  # 7728, 256, 2
    f_mean, f_std = bj.mean(), bj.std()
    bj = (bj - f_mean) / f_std

    taxi = np.stack((taxi_pick, taxi_drop), axis=2)  # 4272, 196, 2
    s_mean, s_std = taxi.mean(), taxi.std()
    taxi = (taxi- s_mean)/s_std

    print('bj.shape:', bj.shape)
    print('taxi.shape:', taxi.shape)

    flow = np.concatenate((bj, taxi), axis=1)  # 4368, 516, 2
    print('flow.shape:', flow.shape)

    train_rate, val_rate = int(args.train_rate), int(args.val_rate)
    train, valid, test = flow[0:-train_rate, :, :], flow[-train_rate:-val_rate, :, :], flow[-val_rate:, :, :]

    # 转变为 torch格式
    trainX, trainY = look_back(train, args.input_dim, args.output_dim)
    validX, validY = look_back(valid, args.input_dim, args.output_dim)
    testX, testY = look_back(test, args.input_dim, args.output_dim)

    log_string(log, f'trainX: {trainX.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validX.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:   {testY.shape}')

    train = TensorDataset(trainX, trainY)
    valid = TensorDataset(validX, validY)
    test = TensorDataset(testX, testY)
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_en = 0

    return train_loader, valid_loader, test_loader, f_mean, f_std, s_mean, s_std, nodes, f_nodes, s_nodes, matrix, log

def d_bjbike(args, log):
    bj_in = pd.read_csv(r'data\beijing\bj_in.csv', index_col=None, header=None).values  # L, N
    bj_out = pd.read_csv(r'data\beijing\bj_out.csv', index_col=None, header=None).values

    bike_pick = pd.read_csv(r'data\nycbike\bike_pick.csv', index_col=None, header=None).values  # L, N
    bike_drop = pd.read_csv(r'data\nycbike\bike_drop.csv', index_col=None, header=None).values
    matrix = pd.read_csv(r'data\bjbike.csv', index_col=None, header=None).values

    f_nodes, s_nodes = len(bj_in[0]), len(bike_pick[0])
    nodes = f_nodes + s_nodes

    len_bike = len(bike_pick)

    bj = np.stack((bj_in, bj_out), axis=2)  # 7728, 256, 2
    f_mean, f_std = bj.mean(), bj.std()
    bj = (bj - f_mean) / f_std

    bike = np.stack((bike_pick, bike_drop), axis=2)  # 4272, 196, 2
    s_mean, s_std = bike.mean(), bike.std()
    bike = (bike - s_mean) / s_std


    print('bj.shape:', bj.shape)
    print('bike.shape:', bike.shape)

    flow = np.concatenate((bj, bike), axis=1)  # 4368, 516, 2
    print('flow.shape:', flow.shape)

    train_rate, val_rate = int(args.train_rate), int(args.val_rate)
    train, valid, test = flow[0:-train_rate, :, :], flow[-train_rate:-val_rate, :, :], flow[-val_rate:, :, :]

    # 转变为 torch格式
    trainX, trainY = look_back(train, args.input_dim, args.output_dim)
    validX, validY = look_back(valid, args.input_dim, args.output_dim)
    testX, testY = look_back(test, args.input_dim, args.output_dim)

    log_string(log, f'trainX: {trainX.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validX.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:   {testY.shape}')

    train = TensorDataset(trainX, trainY)
    valid = TensorDataset(validX, validY)
    test = TensorDataset(testX, testY)
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, f_mean, f_std, s_mean, s_std, nodes, f_nodes, s_nodes, matrix, log
    # return train_loader, valid_loader, test_loader, scaler, nodes, f_nodes, s_nodes, matrix, log


def s_bj(args, log):
    bj_in = pd.read_csv(r'data\beijing\bj_in.csv', index_col=None, header=None).values  # L, N
    bj_out = pd.read_csv(r'data\beijing\bj_out.csv', index_col=None, header=None).values
    matrix = pd.read_csv(r'data\beijing\bj_matrix.csv', index_col=None, header=None).values

    nodes = len(bj_in[0])
    flow = np.stack((bj_in, bj_out), axis=2)  # 4368, 256, 2

    train_rate, val_rate = int(args.train_rate), int(args.val_rate)
    train, valid, test = flow[0:-train_rate, :, :], flow[-train_rate:-val_rate, :, :], flow[-val_rate:, :, :]

    mean, std = train.mean(), train.std()
    train, val, test = (train -mean)/std, (valid -mean)/std, (test -mean)/std

    # 转变为 torch格式
    trainX, trainY = look_back(train, args.input_dim, args.output_dim)
    validX, validY = look_back(valid, args.input_dim, args.output_dim)
    testX, testY = look_back(test, args.input_dim, args.output_dim)

    log_string(log, f'trainX: {trainX.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validX.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:   {testY.shape}')

    train = TensorDataset(trainX, trainY)
    valid = TensorDataset(validX, validY)
    test = TensorDataset(testX, testY)
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, mean, std, nodes, matrix, log

def s_taxi(args, log):

    taxi_pick = pd.read_csv(r'data\nyctaxi\taxi_pick.csv', index_col=None, header=None).values
    taxi_drop = pd.read_csv(r'data\nyctaxi\taxi_drop.csv', index_col=None, header=None).values  # L, N
    matrix = pd.read_csv(r'data\nyctaxi\dis_tt.csv', index_col=None, header=None).values

    nodes = len(taxi_pick[0])
    flow = np.stack((taxi_pick, taxi_drop), axis=2)  # 4368, 256, 2

    train_rate, val_rate = int(args.train_rate), int(args.val_rate)
    train, valid, test = flow[0:-train_rate, :, :], flow[-train_rate:-val_rate, :, :], flow[-val_rate:, :, :]

    mean, std = train.mean(), train.std()
    train, val, test = (train -mean)/std, (valid -mean)/std, (test -mean)/std

    # 转变为 torch格式
    trainX, trainY = look_back(train, args.input_dim, args.output_dim)
    validX, validY = look_back(valid, args.input_dim, args.output_dim)
    testX, testY = look_back(test, args.input_dim, args.output_dim)

    log_string(log, f'trainX: {trainX.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validX.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:   {testY.shape}')

    train = TensorDataset(trainX, trainY)
    valid = TensorDataset(validX, validY)
    test = TensorDataset(testX, testY)
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, mean, std, nodes, matrix, log

def s_bike(args, log):

    bike_pick = pd.read_csv(r'data\nycbike\bike_pick.csv', index_col=None, header=None).values
    bike_drop = pd.read_csv(r'data\nycbike\bike_drop.csv', index_col=None, header=None).values  # L, N
    matrix = pd.read_csv(r'data\nycbike\dis_bb.csv', index_col=None, header=None).values

    nodes = len(bike_pick[0])
    flow = np.stack((bike_pick, bike_drop), axis=2)  # 4368, 256, 2

    train_rate, val_rate = int(args.train_rate), int(args.val_rate)
    train, valid, test = flow[0:-train_rate, :, :], flow[-train_rate:-val_rate, :, :], flow[-val_rate:, :, :]


    mean, std = train.mean(), train.std()
    train, valid, test = (train -mean)/std, (valid -mean)/std, (test -mean)/std

    # 转变为 torch格式
    trainX, trainY = look_back(train, args.input_dim, args.output_dim)
    validX, validY = look_back(valid, args.input_dim, args.output_dim)
    testX, testY = look_back(test, args.input_dim, args.output_dim)

    log_string(log, f'trainX: {trainX.shape}\t\ttrainY: {trainY.shape}')
    log_string(log, f'validX:   {validX.shape}\t\tvalidY:     {validY.shape}')
    log_string(log, f'testX:  {testX.shape}\t\ttestY:   {testY.shape}')

    train = TensorDataset(trainX, trainY)
    valid = TensorDataset(validX, validY)
    test = TensorDataset(testX, testY)
    train_loader = DataLoader(dataset=train, batch_size=args.batch_size, shuffle=False, num_workers=0)
    valid_loader = DataLoader(dataset=valid, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset=test, batch_size=args.batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader, mean, std, nodes, matrix, log

def evalution(out, tgt):

    out = out.reshape(-1)
    tgt = tgt.reshape(-1)
    mae = mean_absolute_error(tgt,out)
    mse = mean_squared_error(tgt,out)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(tgt,out)[0][1]

    return mae, rmse, pcc

# metric
def metrics(pred, label):
    mae = mae_(pred, label)
    mse = mse_(pred, label)
    rmse = rmse_(pred, label)
    pcc = pcc_(pred, label)
    return mae, rmse, pcc

def mae_(pred, label):
    loss = torch.abs(pred - label).type(torch.float32)
    return loss.mean()

def mse_(pred, label):
    loss = (pred-label).type(torch.float32)**2
    return loss.mean()

def rmse_(pred, label):
    loss = torch.sqrt(mse_(pred, label))
    return loss

def pcc_(pred, label):
    label = label.reshape(-1)
    pred = pred.reshape(-1)
    pcc = torch.corrcoef(label, pred)
    return pcc


