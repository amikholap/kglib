import math

from sklearn.metrics import mean_squared_error


def rmse(y_true, y_pred):
    return math.sqrt(mean_squared_error(y_true, y_pred))
