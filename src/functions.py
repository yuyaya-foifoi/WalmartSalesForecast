import numpy as np
import pandas as pd

class Functions:
    def __init__(self):
        pass
    
    @staticmethod
    def appendHolidayFlag(df):
        
        '''assigning features related to holidays
        Args:
            df : DataFrame
        Return:
            DataFrame
        
        '''
        
        from datetime import datetime
        df['SuperBowl'] = np.where(
        (df['Date'] == datetime(2010,2,10)) | (df['Date'] == datetime(2011,2,11)) | 
        (df['Date'] == datetime(2012,2,10)) | (df['Date'] == datetime(2013,2,8)), 1, 0)
        
        df['LaborDay'] = np.where(
        (df['Date'] == datetime(2010,9,10)) | (df['Date'] == datetime(2011,9,9)) | 
        (df['Date'] == datetime(2012,9,7)) | (df['Date'] == datetime(2013,9,6)), 1, 0)
        
        df['ThanksGiving'] = np.where(
        (df['Date']==datetime(2010, 11, 26)) | (df['Date']==datetime(2011, 11, 25)) | 
        (df['Date']==datetime(2012, 11, 23)) | (df['Date']==datetime(2013, 11, 29)), 1, 0)
        
        df['Christmas'] = np.where(
        (df['Date']==datetime(2010, 12, 31)) | (df['Date']==datetime(2011, 12, 30)) | 
        (df['Date']==datetime(2012, 12, 28)) | (df['Date']==datetime(2013, 12, 27)), 1, 0)
        
        df['BlackFriday'] = np.where(
        (df['Date']==datetime(2010, 11, 26)) | (df['Date']==datetime(2011, 11, 25))  | 
        (df['Date']==datetime(2012, 11, 23)) | (df['Date']==datetime(2013, 11, 29)), 1, 0)
        
        df['PreChristmas'] = np.where(
        (df['Date']==datetime(2010, 12, 23)) | (df['Date']==datetime(2010, 12, 24)) | 
        (df['Date']==datetime(2011, 12, 23)) | (df['Date']==datetime(2011, 12, 24)) | 
        (df['Date']==datetime(2012, 12, 23)) | (df['Date']==datetime(2012, 12, 24))  |
        (df['Date']==datetime(2013, 12, 23)) | (df['Date']==datetime(2013, 12, 24)), 1, 0)
        
        return df
    
    def addDateFeature(df, dummy=False):
        
        '''assigning features related to dates
        Args:
            df : DataFrame
        Return:
            DataFrame
        
        '''
        
        import pendulum
        df['Date'] = pd.to_datetime(df['Date'])
        df['WeekofMonth'] = df['Date'].apply(lambda x: pendulum.parse(x).week_of_month)
        df['WeekofYear'] = df['Date'].apply(lambda x: pendulum.parse(x).week_of_year)
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.week
        df['Day'] = df['Date'].dt.day
        
        if dummy:
            df = pd.get_dummies(df, columns=["Month"])
            df = pd.get_dummies(df, columns=["Month"])
            df = pd.get_dummies(df, columns=["Week"])
            df = pd.get_dummies(df, columns=["Week"])
            df = pd.get_dummies(df, columns=["Day"])
            df = pd.get_dummies(df, columns=["Day"])
        
        return
    
    @staticmethod
    def getColumnsDiff(A, B):
        
        '''get diff of columns between df A and df B
        Args:
            A : DataFrame
            B : DataFrame
        Return:
            list
        
        '''
        
        return list(set(A.columns) - set(B.columns))
    
    
    @staticmethod
    def datetimeConverter(df, col_name):
        
        '''convert specific col to datetime
        Args:
            df : DataFrame
            col_name : str
        Return:
            pd.Series
        
        '''
        
        df[col_name] = pd.to_datetime(df[col_name])
        return df[col_name]