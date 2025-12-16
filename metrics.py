import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score

def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    rmse_mean = rmse / (abs(np.mean(y_true)) + 1e-8)
    rmse_std  = rmse / (np.std(y_true) + 1e-8)
    r2 = r2_score(y_true, y_pred)

    y_true_bin = (y_true > 0).astype(int).flatten()
    y_pred_bin = (y_pred > 0).astype(int).flatten()

    acc  = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    rec  = recall_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    f1   = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)

    return mse, rmse_mean, rmse_std, r2, acc, prec, rec, f1
