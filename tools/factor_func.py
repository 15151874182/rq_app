import os,sys,time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from xtquant import xtdata

class Factor(): ##因子库
    
    def __init__(self):
        pass

###############时间类因子
    @staticmethod
    def dayofweek(data):
        data['dayofweek']=data.index.to_series().apply(lambda x : x.dayofweek / 6 - 0.5)
        return data
    
    @staticmethod
    def dayofmonth(data):
        data['dayofmonth']=data.index.to_series().apply(lambda x : (x.day-1) / 30 - 0.5)
        return data
    
    @staticmethod
    def dayofyear(data):
        data['dayofyear']=data.index.to_series().apply(lambda x : (x.dayofyear-1) / 365 - 0.5)
        return data
    
    @staticmethod
    def weekofyear(data):
        data['weekofyear']=data.index.to_series().apply(lambda x : (x.weekofyear-1) / 52 - 0.5)
        return data
    
    @staticmethod
    def monthofyear(data):
        data['monthofyear']=data.index.to_series().apply(lambda x : (x.month-1) / 11 - 0.5)
        return data
        
###############技术面因子            
    @staticmethod
    def return_n(data,ns=[1,2,3,5,7,10,20,30]):
        for n in ns:
            data[f'return_{n}']=data['return'].shift(n)  
        return data
    
    @staticmethod
    def momentum_n(data,ns=[21]):
        for n in ns:
            data[f'momentum_{n}']=data['close']-data['close'].shift(n)   
        return data
    
    @staticmethod
    def ma_n(data,ns=[1,2,3,5,7,10,20,30]):####SMA(Simple Moving Average)
        for n in ns:
            data[f'sma_{n}']=data['close'].rolling(n).mean()
        return data
    
    @staticmethod
    def ema_n(data,ns=[1,2,3,5,7,10,20,30]):####EMA(Exponential Moving Average)
        for n in ns:
            data[f'ema_{n}']=data['close'].ewm(n, adjust=False).mean()
        return data
    
    @staticmethod
    def rsi(data):####RSI(Relative Strength Index)
        delta = data['close']-data['preclose']
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / avg_loss
        data['rsi'] = 100 - (100 / (1 + rs))
        return data
    
    @staticmethod
    def macd(data):
        ema12 = data['close'].ewm(span=12, min_periods=1, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, min_periods=1, adjust=False).mean()
        dif = ema12 - ema26
        # dea = dif.ewm(span=9, min_periods=1, adjust=False).mean()  
        # data['macd']=2*(dif-dea)
        data['macd']=dif
        return data

    @staticmethod
    def cci(data):####CCI      
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        sma_typical_price = typical_price.rolling(window=5).mean()
        mean_deviation = typical_price.rolling(window=5).apply(lambda x: abs(x - x.mean()).mean(), raw=True)
        data['cci'] = (typical_price - sma_typical_price) / (0.015 * mean_deviation)    
        return data
    
    @staticmethod
    def yz_estimator(data):  
        ####YZ estimator
        n=2 ##The highest efficiency is reached when n=2,a=1.34
        k=0.34/(1.34+(n+1)/(n-1)) ##constant k 
        data['oi']=np.log(data['open']/data['close'].shift(1)) ## the normalized open
        data['ci']=np.log(data['close']/data['open']) ##the normalized close
        data['ui']=np.log(data['high']/data['open']) ##the normalized high
        data['di']=np.log(data['low']/data['open']) ## the normalized low    
        
        o1=data['oi'].shift(1)
        o2=data['oi']
        c1=data['ci'].shift(1)
        c2=data['ci']
        u1=data['ui'].shift(1)
        u2=data['ui']
        d1=data['di'].shift(1)
        d2=data['di']
        
        Vo=0.5*(o1-o2)**2 ##unbias variance of normalized open
        Vc=0.5*(c1-c2)**2 ##unbias variance of normalized close
        VRs=0.5*(u1**2+d1**2-u1*c1-d1*c1+   ##VRs variance found by Rogers and Satchell
                       u2**2+d2**2-u2*c2-d2*c2)
        data['yz']=Vo+k*Vc+(1-k)*VRs ##YZ estimator
        return data
    
    @staticmethod
    def atr(data):####ATR(Average True Range)          
        
        data['h-l']  = abs(data['high']-data['low'])
        data['h-pc'] = abs(data['high']-data['preclose'])
        data['l-pc'] = abs(data['low']-data['preclose'])
        data['tr']   = data[['h-l','h-pc','l-pc']].max(axis=1,skipna=False)
        data['atr']  = data['tr'].rolling(21).mean()
        data = data.drop(['h-l', 'h-pc', 'l-pc', 'tr'], axis=1)
        return data
    
    @staticmethod
    def obv(data):####ATR(Average True Range)                  
        #### OBV(On-Balance Volume)
        data['price_change'] = data['close'].diff()
        ##put 'volume' into positive or negtive based on Price_Change
        data['obv'] = data.apply(lambda row: row['volume'] if row['price_change'] > 0 else
                                 (-1) * row['volume'] if row['price_change'] < 0 else 0, axis=1)
        data['obv'] = data['obv'].cumsum() ##cumulative sum
        del data['price_change']
        return data
    
    @staticmethod
    def vwma(data):#### VWMA(Volume Weighted Moving Average)        
        data['vwma'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()        
        # data['vwma2'] = (data['close'] * data['volume']).rolling(21).sum() / data['volume'].rolling(21).sum()      
        return data
    
    @staticmethod
    def boll_band(data): #### Bollinger Bands       
        window = 3  # Rolling window size
        num_std = 2  # Number of standard deviations for bands
        rolling_mean = data['close'].rolling(window=window).mean()
        rolling_std = data['close'].rolling(window=window).std()
        # Calculate upper and lower Bollinger Bands
        data['upper_band'] = rolling_mean + (rolling_std * num_std)
        data['lower_band'] = rolling_mean - (rolling_std * num_std)
        return data

    @staticmethod
    def kdj(data, n=9, m=3):
        """
        生成KDJ指标（随机指标）
        参数说明：
        data: 必须包含high、low、close三列的DataFrame
        n: 计算RSV的周期（默认9天）
        m: 计算K/D值的平滑周期（默认3天）
        """
        # 计算n日最低价和最高价
        low_min = data['low'].rolling(window=n, min_periods=1).min()
        high_max = data['high'].rolling(window=n, min_periods=1).max()
        # 计算RSV（未成熟随机值）
        data['rsv'] = (data['close'] - low_min) / (high_max - low_min + 1e-8) * 100  # +1e-8避免除零
        # 计算K值（RSV的m日SMA）
        data['k_line'] = data['rsv'].ewm(alpha=1/m, adjust=False).mean()  # 指数平滑更符合传统算法
        # 计算D值（K值的m日SMA）
        data['d_line'] = data['k_line'].ewm(alpha=1/m, adjust=False).mean()
        # 计算J值（3K - 2D）
        data['j_line'] = 3 * data['k_line'] - 2 * data['d_line']
        # 删除中间列
        data.drop('rsv', axis=1, inplace=True)
        
        return data
    
    @staticmethod
    def derivative(data,col): #### 某列计算 一阶导数+二阶导数
        # 计算一阶导数
        data[f'{col}_1st_derivative']= data[col].diff()
        # 计算二阶导数
        data[f'{col}_2nd_derivative']= data[f'{col}_1st_derivative'].diff()  
        return data

    @staticmethod
    def pe(data,id): 
        def func1(x):
            if x=='0331':
                return 1
            elif x=='0630':
                return 2
            elif x=='0930':
                return 3
            elif x=='1231':
                return 4
        ######pe
        try:
            finan_data=xtdata.get_financial_data([id], table_list=['Income','Capital','CashFlow','Balance'],
                                          start_time=data.index[0].strftime('%Y-%m-%d').replace('-',''),
                                          end_time=data.index[-1].strftime('%Y-%m-%d').replace('-',''),
                                          report_type='report_time')
            
            net_profit = finan_data[id]['Income'][['m_timetag','net_profit_incl_min_int_inc']]#净利润
            net_profit = net_profit.sort_values(by='m_timetag')
            net_profit = net_profit.drop_duplicates(subset = 'm_timetag',keep = 'last')[['m_timetag','net_profit_incl_min_int_inc']]
            net_profit=net_profit.set_index('m_timetag')
            full_timetag=[y+m for y in [str(year) for year in range(data.index[0].year, data.index[-1].year+1)] for m in ['0331','0630','0930','1231']]
            net_profit=net_profit.reindex(full_timetag)
            net_profit.columns=['net_profit']
            net_profit['year']=net_profit.index.to_series().apply(lambda x: x[:4])

            net_profit['season']=net_profit.index.to_series().apply(lambda x: func1(x[4:]))
            
            net_profit['single_profit']=np.nan
            season_col=net_profit.columns.get_loc("season")
            net_profit_col=net_profit.columns.get_loc("net_profit")
            single_profit_col=net_profit.columns.get_loc("single_profit")
            
            for i in range(len(net_profit)):     
                if net_profit.iloc[i,season_col]==1:
                    net_profit.iloc[i,single_profit_col]=net_profit.iloc[i,net_profit_col]
                else:
                    net_profit.iloc[i,single_profit_col]=net_profit.iloc[i,net_profit_col]-net_profit.iloc[i-1,net_profit_col]
            net_profit["single_profit"] = net_profit.groupby("season")["single_profit"].apply(lambda group: group.ffill()) ##nan拿历史同期填补
            net_profit["net_profit_fix"] =net_profit["single_profit"].rolling(4).sum()
            net_profit=net_profit.dropna(subset='net_profit_fix')
            net_profit=net_profit[['net_profit_fix']]
            net_profit.index = pd.to_datetime(net_profit.index, format='%Y%m%d')
            data = pd.merge_asof(data, net_profit, left_index=True, right_index=True, direction='forward')
            
            captital=finan_data[id]['Capital'][['m_timetag','total_capital']]
            captital = captital.sort_values(by='m_timetag')
            captital = captital.drop_duplicates(subset = 'm_timetag',keep = 'last')[['m_timetag','total_capital']]
            captital=captital.set_index('m_timetag')     
            captital.index=captital.index.to_series().apply(lambda x: datetime.strptime(x, '%Y%m%d'))
            data = pd.merge_asof(data, captital, left_index=True, right_index=True, direction='forward')
            data['total_capital']=data['total_capital'].ffill()
            data['market_value']=data['total_capital']*data['close']
            
            data['pe']=data['market_value']/data['net_profit_fix']
            data['ep']=1/data['pe']
        except:
            data['market_value']=np.nan
            data['net_profit_fix']=np.nan
            data['pe']=np.nan
            data['ep']=np.nan
        del data['market_value'],data['net_profit_fix']
        return data
    
    @staticmethod
    def ps(data,id): 
        def func1(x):
            if x=='0331':
                return 1
            elif x=='0630':
                return 2
            elif x=='0930':
                return 3
            elif x=='1231':
                return 4
        ######pe
        try:
            finan_data=xtdata.get_financial_data([id], table_list=['Income','Capital','CashFlow','Balance'],
                                          start_time=data.index[0].strftime('%Y-%m-%d').replace('-',''),
                                          end_time=data.index[-1].strftime('%Y-%m-%d').replace('-',''),
                                          report_type='report_time')
            revenue = finan_data[id]['Income'][['m_timetag','revenue_inc']]#营业收入
            revenue = revenue.sort_values(by='m_timetag')
            revenue = revenue.drop_duplicates(subset = 'm_timetag',keep = 'last')[['m_timetag','revenue_inc']]
            revenue=revenue.set_index('m_timetag')
            full_timetag=[y+m for y in [str(year) for year in range(data.index[0].year, data.index[-1].year+1)] for m in ['0331','0630','0930','1231']]
            revenue=revenue.reindex(full_timetag)
            revenue.columns=['revenue']
            revenue['year']=revenue.index.to_series().apply(lambda x: x[:4])
            revenue['season']=revenue.index.to_series().apply(lambda x: func1(x[4:]))
            
            revenue['single_revenue']=np.nan
            season_col=revenue.columns.get_loc("season")
            revenue_col=revenue.columns.get_loc("revenue")
            single_revenue_col=revenue.columns.get_loc("single_revenue")
            
            for i in range(len(revenue)):     
                if revenue.iloc[i,season_col]==1:
                    revenue.iloc[i,single_revenue_col]=revenue.iloc[i,revenue_col]
                else:
                    revenue.iloc[i,single_revenue_col]=revenue.iloc[i,revenue_col]-revenue.iloc[i-1,revenue_col]
            revenue["single_revenue"] = revenue.groupby("season")["single_revenue"].apply(lambda group: group.ffill()) ##nan拿历史同期填补
            revenue["revenue_fix"] =revenue["single_revenue"].rolling(4).sum()
            revenue=revenue.dropna(subset='revenue_fix')
            revenue=revenue[['revenue_fix']]
            revenue.index = pd.to_datetime(revenue.index, format='%Y%m%d')
            data = pd.merge_asof(data, revenue, left_index=True, right_index=True, direction='forward')

            captital=finan_data[id]['Capital'][['m_timetag','total_capital']]
            captital = captital.sort_values(by='m_timetag')
            captital = captital.drop_duplicates(subset = 'm_timetag',keep = 'last')[['m_timetag','total_capital']]
            captital=captital.set_index('m_timetag')     
            captital.index=captital.index.to_series().apply(lambda x: datetime.strptime(x, '%Y%m%d'))
            data = pd.merge_asof(data, captital, left_index=True, right_index=True, direction='forward')
            data['total_capital']=data['total_capital'].ffill()
            data['market_value']=data['total_capital']*data['close']
            
            data['ps']=data['market_value']/data['revenue_fix']
            data['sp']=1/data['ps']
        except:
            data['market_value']=np.nan
            data['revenue_fix']=np.nan
            data['ps']=np.nan
            data['sp']=np.nan            
        del data['market_value'],data['revenue_fix']            
        return data
    
    @staticmethod
    def pb(data,id): 
        def func1(x):
            if x=='0331':
                return 1
            elif x=='0630':
                return 2
            elif x=='0930':
                return 3
            elif x=='1231':
                return 4
        ######pe
        try:
            finan_data=xtdata.get_financial_data([id], table_list=['Income','Capital','CashFlow','Balance'],
                                          start_time=data.index[0].strftime('%Y-%m-%d').replace('-',''),
                                          end_time=data.index[-1].strftime('%Y-%m-%d').replace('-',''),
                                          report_type='report_time')
            asset_debt = finan_data[id]['Balance'][['m_timetag','tot_assets','tot_liab']]#资产和负债
            asset_debt=asset_debt.set_index('m_timetag')
            net_asset=asset_debt['tot_assets']-asset_debt['tot_liab']
            net_asset=net_asset.reset_index()
            net_asset.columns=['m_timetag','net_asset_inc']
            
            net_asset = net_asset.sort_values(by='m_timetag')
            net_asset = net_asset.drop_duplicates(subset = 'm_timetag',keep = 'last')[['m_timetag','net_asset_inc']]
            net_asset=net_asset.set_index('m_timetag')
            full_timetag=[y+m for y in [str(year) for year in range(data.index[0].year, data.index[-1].year+1)] for m in ['0331','0630','0930','1231']]
            net_asset=net_asset.reindex(full_timetag)
            net_asset.columns=['net_asset']
            net_asset['year']=net_asset.index.to_series().apply(lambda x: x[:4])
            net_asset['season']=net_asset.index.to_series().apply(lambda x: func1(x[4:]))
            
            net_asset['single_net_asset']=np.nan
            season_col=net_asset.columns.get_loc("season")
            net_asset_col=net_asset.columns.get_loc("net_asset")
            single_net_asset_col=net_asset.columns.get_loc("single_net_asset")
            
            for i in range(len(net_asset)):     
                if net_asset.iloc[i,season_col]==1:
                    net_asset.iloc[i,single_net_asset_col]=net_asset.iloc[i,net_asset_col]
                else:
                    net_asset.iloc[i,single_net_asset_col]=net_asset.iloc[i,net_asset_col]-net_asset.iloc[i-1,net_asset_col]
            net_asset["single_net_asset"] = net_asset.groupby("season")["single_net_asset"].apply(lambda group: group.ffill()) ##nan拿历史同期填补
            net_asset["net_asset_fix"] =net_asset["single_net_asset"].rolling(4).sum()
            net_asset=net_asset.dropna(subset='net_asset_fix')
            net_asset=net_asset[['net_asset_fix']]
            net_asset.index = pd.to_datetime(net_asset.index, format='%Y%m%d')
            data = pd.merge_asof(data, net_asset, left_index=True, right_index=True, direction='forward')

            captital=finan_data[id]['Capital'][['m_timetag','total_capital']]
            captital = captital.sort_values(by='m_timetag')
            captital = captital.drop_duplicates(subset = 'm_timetag',keep = 'last')[['m_timetag','total_capital']]
            captital=captital.set_index('m_timetag')     
            captital.index=captital.index.to_series().apply(lambda x: datetime.strptime(x, '%Y%m%d'))
            data = pd.merge_asof(data, captital, left_index=True, right_index=True, direction='forward')
            data['total_capital']=data['total_capital'].ffill()
            data['market_value']=data['total_capital']*data['close']            

            data['pb']=data['market_value']/data['net_asset_fix']
            data['bp']=1/data['pb']
        except:
            data['market_value']=np.nan
            data['net_asset_fix']=np.nan
            data['pb']=np.nan
            data['bp']=np.nan            
        del data['market_value'],data['net_asset_fix']            
        return data
    
    @staticmethod
    def select_feas(data,threshold=0.95): #### data里要包含y，通过相关性矩阵自动筛选特征，去除多重共线性
        corr=data.corr()
        corr['y']=corr['y'].apply(lambda x:abs(x)) ##相关性用绝对值
        corr=corr.sort_values('y',ascending=False)
        feas_to_del=[]
        length=len(corr)
        for i in range(1,length):
            df=corr.iloc[i,:]
            if df.name not in feas_to_del: ##如果已经在待删除的list中，不需要再计算
                df=df[df!=1] ##自己和自己相关性=1，不需要算在内
                bad_feas=list(df[df>=threshold].index) ##bad_feas判断依据就是和当前高coe特征的共线性太强，没有存在必要
                if bad_feas: ##存在bad_feas
                    feas_to_del+=bad_feas
        
        feas_selected=[fea for fea in list(data.columns) if fea not in feas_to_del]
        data_selected=data[feas_selected]
        return feas_to_del,data_selected ##返回待删除feas列表和筛选后对的data
            
        
    