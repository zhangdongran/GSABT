from sklearn.metrics import mean_squared_error,mean_absolute_error
import numpy as np


def evalution(tgt,out):
    tgt = tgt.reshape(-1)
    out = out.reshape(-1)
    mae = mean_absolute_error(tgt,out)
    mse = mean_squared_error(tgt,out)
    rmse = np.sqrt(mse)
    pcc = np.corrcoef(tgt,out)[0][1]

    return rmse, mae, pcc