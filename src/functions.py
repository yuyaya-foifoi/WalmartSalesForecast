import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
        df['WeekofMonth'] = df['Date'].apply(lambda x: pendulum.parse(x).week_of_month)
        df['WeekofYear'] = df['Date'].apply(lambda x: pendulum.parse(x).week_of_year)
        df['Date'] = pd.to_datetime(df['Date'])
        #df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.week
        df['Day'] = df['Date'].dt.day
        
        if dummy:
            df = pd.get_dummies(df, columns=["WeekofMonth"])
            df = pd.get_dummies(df, columns=["WeekofYear"])
            df = pd.get_dummies(df, columns=["Date"])
            df = pd.get_dummies(df, columns=["Month"])
            df = pd.get_dummies(df, columns=["Day"])
        
        return df
    
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
    
    @staticmethod
    def rollingInterpolation(features, roll=16):
        
        '''interpolating CPI and Unemployment by rolling
        Args:
            features : DataFrame
            roll : int
        Return:
            DataFrame
        
        '''
        
        stores_list = list(range(1,46))
        for store in stores_list:
            tmp = features[features['Store'] == store]
            tmp.Date = pd.to_datetime(tmp.Date)
            tmp = tmp.set_index('Date')

            features.loc[(features.Store == store),['CPI']] = list(tmp.CPI.fillna(tmp.CPI.rolling(roll,min_periods=1).mean()))
            features.loc[(features.Store == store),['Unemployment']] = list(tmp.Unemployment.fillna(tmp.Unemployment.rolling(roll,min_periods=1).mean()))
            
        return features
    
    
    def createLagFeatures(df, gp_cols=['Store','Dept'], target='Weekly_Sales', lags=[52, 104]):
        
        '''create lag feature
        Args:
            df : DataFrame
            gp_cols : list
            target : str
            lags : list
        Return:
            DataFrame
        
        '''
        
        gp = df.groupby(gp_cols)
        for i in lags:
            df['_'.join([target, 'LagFeat', str(i)])] = gp[target].shift(i).values

        return df
    
    
    def createDateStatsFeatures(df, variable, gp_cols, target='Weekly_Sales', funcs=['mean','median','max','min','std','sum']):
        
        '''create data aggregate features
        Args:
            df : DataFrame
            variable : str
            gp_cols : list
            target : str
            funcs : list
        Return:
            DataFrame
        
        '''
        
        train_df = df.loc[~(df.train_or_test=='test'), :]
        gp = train_df.groupby(gp_cols)
        newdf = df[gp_cols].drop_duplicates().reset_index(drop=True)
        for func in funcs:
            tmp = gp[target].agg(func).reset_index()
            tmp.rename(columns={target:variable + func}, inplace=True)
            newdf = newdf.merge(tmp, on=gp_cols, how='left')
        return df.merge(newdf, on=gp_cols, how='left')
    
    
    def createClusteredFeatures(df, num_clusters=[3, 5, 10, 15, 20], remove_cols=['Date', 'Store', 'Dept', 'train_or_test']):
        
        '''create data clustered features
        Args:
            df : DataFrame
            num_clusters : list
            remove_cols : list
        Return:
            DataFrame
        
        '''
 
        cols = list(df.columns[df.isnull().any() == False])

        for remove_col in remove_cols:
            cols.remove(remove_col)

        for num_cluster in num_clusters:

            KM = KMeans(n_clusters = num_cluster, random_state = 0, n_jobs = -1)
            KM.fit(df.loc[:, cols])
            km_pred = KM.predict(df.loc[:, cols])
            km_distance = KM.transform(df.loc[:, cols])

            for clst in range(0, num_cluster):
                df['Labels_{nums}_{clst}_distance'.format(nums=num_cluster, clst=clst)] = km_distance[:, clst]

            df['Labels_{}'.format(num_cluster)] = km_pred
            df =  Functions.createDateStatsFeatures(df, 'Labels_{}_Sales_'.format(num_cluster), ['Store','Dept', 'Labels_{}'.format(num_cluster)])

            unique, count = np.unique(km_pred, return_counts=True)
            print('-'*10, num_cluster, ' cluster', '-'*10)
            for i in range(0, len(count)):
                print('Cluster : ', unique[i], 'Nums : ', count[i])
                
        return df