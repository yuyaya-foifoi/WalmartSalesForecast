import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Modeling:
    def __init__(self):
        pass
    
    @staticmethod
    def WMAE(X, ans, pred): 
        '''compute WMAE
        Args:
            X : DataFrame
            ans : Series
            pred : Series
        Return:
            np.array
        
        '''
        weights = X['IsHoliday'].apply(lambda is_holiday:5 if   is_holiday else 1)
        error = np.sum(weights * np.abs(ans - pred), axis=0) / np.sum(weights)
        return error
    
    @staticmethod
    def computeMetrics(X_test, y_test, y_pred):
        '''compute Loss
        Args:
            X_test : DataFrame
            y_test : Series
            y_pred : Series
        
        '''
        print("MAE: ", metrics.mean_absolute_error(y_test, y_pred)) 
        print("MSE: ", metrics.mean_squared_error(y_test, y_pred)) 
        print("RMSE: ", np.sqrt(metrics.mean_squared_error(y_test, y_pred))) 
        print("R^2:", r2_score(y_test, y_pred))
        print('WMAE: ', Modeling.WMAE(X_test, y_test, y_pred))
    
