import os,sys,time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import traceback
from copy import deepcopy 
import gc

from xtquant.xttrader import XtQuantTrader, XtQuantTraderCallback
from xtquant.xttype import StockAccount
from xtquant import xtconstant
from xtquant import xtdata

import baostock as bs
from config.config import ConfigParser,stock_info,stock_info_path
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
from tools.factor_func import Factor
from tools.download_func import Download
from tools.analysis_func import Analysis
from tools.analysis_func import Metrics
from tools.general_func import General
from tools.convert_func import Convert  
from tools.alpha101 import Alphas101
from tools.alpha191 import Alphas191  
from sklearn.model_selection import train_test_split
from lightgbm import  plot_tree
from model import MyLGB
from model import Model,ModelCombiner
from imblearn.over_sampling import SMOTE

class Agent():
    
    def __init__(self,args):
        self.pkl_suffix=f'_seq_len{args.seq_len}_{args.label}.pkl'
        self.csv_suffix=f'_seq_len{args.seq_len}_{args.label}.csv'
        
    def read_raw_data(self,config):
        data = xtdata.get_market_data_ex([],[config.id],period='1d',
                                         start_time = '20100101',count=-1,
                                         dividend_type='front_ratio')
        df=data[config.id]
        df.index=pd.Series(df.index).apply(lambda x:pd.to_datetime(x, format='%Y%m%d'))
        df.rename(columns={'preClose': 'preclose'}, inplace=True)
        df=df[['open', 'high', 'low', 'close', 'preclose','volume', 'amount','suspendFlag']]
        df['return']=df['close']/df['preclose']-1
        return df

    def read_fe_data(self,config):
        data=pd.read_csv(config.fe_data_path,index_col=0,parse_dates=True)
        return data

        
    def make_fe_data(self,config):
        data=self.read_raw_data(config)
        # data.to_csv(config.data_path)    ##保存原始数据
        ##预处理停盘的数据
        data[['volume', 'amount']] = data[['volume', 'amount']].replace(0, None)  # 将 0 替换为 None 以便使用 ffill 和 bfill
        data[['volume', 'amount']] = data[['volume', 'amount']].ffill().bfill()  # 先向前填充，再向后填充
        fe_data=self.feature_engineering(data,config.id)
        fe_data.to_csv(config.fe_data_path)    ##保存FE后的数据
    
    def make_label(self,args,data,seq_len): 
        if args.label=='return': ##绝对收益率
            data['y']=data['close'].shift(-1*seq_len)/data['close']-1
            # data['y']=np.log(1 + abs(data['y']*100))
            # data['y']=data['y'].rank(pct=True)
            # data['y']=np.tanh(data['y']*100)
            # data['y']=np.arctan(data['y']*10)            
            # data=data.dropna() ##shift操作会产生NAN值
        elif args.label=='sharp':
            data['future_return']=data['return'].shift(-1*seq_len)
            def func(df):
                if len(df)!= args.seq_len:
                    return np.nan
                else:
                    return Metrics.sharpe_ratio_annual(df,rf=0.03)
            data['y']=data['future_return'].rolling(args.seq_len).apply(func)
            data=data.dropna() ##shift操作会产生NAN值
        return data

    # def make_label2(self,data,bench_index_data,seq_len):##相对zz500超额收益率
    #     bench_index_data=bench_index_data[['close','atr','macd','vwma']]
    #     bench_index_data.rename(columns={'close': 'bench_index_close',
    #                                      'atr': 'bench_index_atr',
    #                                      'macd': 'bench_index_macd',
    #                                      'vwma': 'bench_index_vwma'}, inplace=True)
    #     data=data.join(bench_index_data)
        
    #     data['atr_ratio']=data['atr']/data['bench_index_atr']
    #     data['macd_ratio']=data['macd']/data['bench_index_macd']
    #     data['vwma_ratio']=data['vwma']/data['bench_index_vwma']
        
    #     data['y']=data['close'].shift(-1*seq_len)/data['close']-1
    #     # data['y']=(data['close'].shift(-1*seq_len)/data['close']-1)-(data['bench_index_close'].shift(-1*seq_len)/data['bench_index_close']-1)
    #     data=data.dropna() ##shift操作会产生NAN值
    #     return data
    
    ## feature_engineering 
    def feature_engineering(self,data,id):
        
        ###############Alphas101
        # alpha=Alphas101(data)
        # strings = [f"{i:03d}" for i in range(1, 102)]
        # for string in strings:
        #     try:
        #         data[f'alpha{string}']=getattr(alpha, f'alpha{string}')()    
        #     except:
        #         # print(f'{string} wrong')
        #         pass
        
        # ###############Alphas191
        # alpha=Alphas191(data)
        # strings = [f"{i:03d}" for i in range(1, 192)]
        # for string in strings:
        #     try:
        #         data[f'gtja_alpha{string}']=getattr(alpha, f'alpha{string}')()
        #         # print(f'{string} done')
        #     except:
        #         # print(f'{string} wrong')  
        #         pass
        
        ###############时间类因子
        # data=Factor.dayofweek(data)
        # data=Factor.dayofmonth(data)
        # data=Factor.dayofyear(data)
        # data=Factor.weekofyear(data)
        # data=Factor.monthofyear(data)
        
        ###############技术面因子  
        # data=Factor.return_n(data)
        # data=Factor.momentum_n(data)
        # data=Factor.ma_n(data)
        # data=Factor.ema_n(data)
        # data=Factor.rsi(data)
        data=Factor.macd(data)
        data=Factor.atr(data) 
        data=Factor.vwma(data)
        data=Factor.pe(data,id)
        data=Factor.ps(data,id)
        data=Factor.pb(data,id)
        # data=Factor.cci(data)
        # data=Factor.yz_estimator(data)
        # data=Factor.obv(data)
        # data=Factor.boll_band(data)
        
        # data=data.interpolate(method='linear', limit_direction='both',axis=0)
        return data
    

    def make_dataset(self,fe_data,fea_select,start_time,end_time,test_start_time):##测试集按固定日期，且保留trainval集用于后面多折交叉验证
        fe_data=fe_data[start_time:end_time] 
        
        trainval=fe_data[:test_start_time]
        test=fe_data[test_start_time:]
        
        trainval = trainval.drop_duplicates() 
        trainval=trainval[trainval['suspendFlag']!=1] ##train去掉停盘的
        
        # x_trainval=trainval[fea_select+['suspendFlag']]
        # y_trainval=trainval['y']
        
        # if len(test[test['suspendFlag']==1]) != 0:
        #     pass
        # test.loc[test['suspendFlag']==1,:]=np.nan
        
        # x_test=test[fea_select]
        # y_test=test['y']
    
        # return x_trainval, x_test, y_trainval, y_test
        return trainval, test

    @staticmethod
    def update_stock_info(args,agent): 
        dfs=[]
        for id in tqdm(args.ids):
            df=[id,'stock',f'data/all/{id}.csv','d',2,f'data/all/{id}_fe.csv',f'model/all/{id}',1,0]
            dfs.append(df)
        dfs=pd.DataFrame(dfs,columns=stock_info.columns)
        new_stock_info=pd.concat([stock_info,dfs],axis=0)
        new_stock_info=new_stock_info.reset_index(drop=True)
        new_stock_info.to_csv(stock_info_path,index=False)

    def update_local_data(self,args,agent):
        
        downloader = Download()
        last_datas=[] ##收集所有id真实的收盘价和return，来验证策略
        failed_ids=[]
        for id in tqdm(args.ids):
            try:
                config=ConfigParser(id) 
                downloader.download_data(config) 
                last_data=pd.read_csv(config.raw_data_path,index_col=0,parse_dates=True).iloc[-1] ##更新同时获取最后一天数据，实盘验证用
                last_datas.append([id,last_data])
                self.make_fe_data(config) ##更新本地fe_data数据
            except:
                traceback.print_exc()   
                failed_ids.append(id)
        ids,ys=zip(*last_datas) ##ys是多个id的预测df
        res=pd.concat(ys,axis=1)
        res.columns=list(ids)
        res=res.loc[['close', 'pctChg']]      
        res.loc['res'] = round(res.loc['close'],2).apply(str) + ' && '+round(res.loc['pctChg'],2).apply(str)+'%'
        res = res.drop(['close','pctChg'], axis=0)
        return failed_ids
    
    # def train(self,args):
    #     trainval_ress=[]  #保存多个id的验证集结果，用于筛选股票或调参
    #     test_ress=[] #保存多个id的测试集结果，用于回测
    #     tmp_ress=[]
    #     failed_ids=[]

    #     for id in tqdm(args.ids): ##聚合所有stocks，使其能计算横截面信息
    #         try:
    #             config=ConfigParser(id)    
    #             fe_data=self.read_fe_data(config)
    #             config2=ConfigParser(args.bench_index_code)    
    #             bench_index_data=self.read_fe_data(config2)
    #             y_fe_data=self.make_label2(fe_data,bench_index_data,seq_len=args.seq_len)
    #             ##预处理停盘的数据
    #             y_fe_data.loc[y_fe_data['volume']==0,:]=np.nan
    #             y_fe_data=y_fe_data.fillna(method='ffill').fillna(method='bfill')               
                
    #             y_fe_data=y_fe_data[args.start_time:]
    #             y_fe_data['id']=id
    #             tmp_ress.append(y_fe_data)    
    #         except:
    #             pass
    #     data=pd.concat(tmp_ress,axis=0)
    #     scaler = RobustScaler()
    #     # scaler = StandardScaler()
    #     # scaler = MinMaxScaler()
        
    #     def scale_group(group):
    #         # group['y'] = scaler.fit_transform(group[['y']])
    #         group['atr_ratio'] = scaler.fit_transform(group[['atr_ratio']])
    #         group['macd_ratio'] = scaler.fit_transform(group[['macd_ratio']])
    #         group['vwma_ratio'] = scaler.fit_transform(group[['vwma_ratio']])
    #         # group['y'] = group[['y']]
    #         return group
    #     data = data.groupby(data.index).apply(scale_group) ##批标准化
        
    #     group=data.groupby('id')
    #     for id,y_fe_data in tqdm(group):

    #         y_fe_data = y_fe_data.drop(columns=['id'])
    #         try: 
    #             x_trainval, x_test, y_trainval, y_test = self.make_dataset2(fe_data=y_fe_data,
    #                                                                               fea_select=args.fea_select,
    #                                                                               start_time=args.start_time,
    #                                                                               end_time=args.end_time,
    #                                                                               test_start_time=args.test_start_time) 
    #             # if len(x_trainval)<1000: ##训练样本过少
    #             #     failed_ids.append(id)
    #             #     continue
                
    #             # 创建模型并训练===========================================================        
    #             mylgb=MyLGB()
                
    #             best_model=mylgb.train(x_trainval, x_test, y_trainval, y_test,method='simple')     
    #             # model_study=Analysis.model_study(best_model)
    #             mylgb.save(best_model,config.model_path+self.pkl_suffix) 
    #             #在训练验证集上的效果
    #             y_trainval_pred=mylgb.predict(best_model, x_trainval)
    #             trainval_res=General.get_res(y_trainval_pred, y_trainval) ##单个id验证集结果
    #             trainval_ress.append([id,trainval_res])
                
    #             #在测试集上的效果
    #             y_test_pred=mylgb.predict(best_model, x_test)
    #             test_res=General.get_res(y_test_pred, y_test) ##单个id测试集结果    
    #             # print('test mse:',mean_squared_error(test_res['gt'],test_res['pred']))
    #             test_ress.append([id,test_res])            
    #         except:
    #             traceback.print_exc()        
    #             failed_ids.append(id)   
                
    #     return trainval_ress,test_ress,failed_ids

    def train(self,args):
        trainval_ress=[]  #保存多个id的验证集结果，用于筛选股票或调参
        test_ress=[] #保存多个id的测试集结果，用于回测
        failed_ids=[]
        gt_pred_corrs=[]
        
        for id in tqdm(args.ids):
            try:
                #读取id对应配置
                config=ConfigParser(id)                   
                #读取fe后的数据
                fe_data=self.read_fe_data(config)
      
                # config2=ConfigParser(args.bench_index_code)    
                # bench_index_data=self.read_fe_data(config2)
                # y_fe_data=self.make_label2(fe_data,bench_index_data,seq_len=args.seq_len)                
               
                #按seq_len长度制作label
                y_fe_data=self.make_label(args,fe_data,seq_len=args.seq_len)
                
                y_fe_data=y_fe_data[args.start_time:]
                feas=deepcopy(args.fea_select)
                
                # if config.selected_feas: ##如果存在筛选好的特征,否则用默认值
                #     args.fea_select=config.selected_feas.split('+')
                
                # path='config/can_use_pe.csv' 
                # df=pd.read_csv(path)
                # can_use_pe_ids=list(df['id'])
                # if id not in can_use_pe_ids: ##有些票没有pe数据，就不用
                #     feas.remove('pe')
                # if y_fe_data["pb"].isna().all(): ##有些票没有pe数据，就不用
                #     feas.remove('pb')
                    
                trainval, test= self.make_dataset(fe_data=y_fe_data,
                                                    fea_select=feas,
                                                    start_time=args.start_time,
                                                    end_time=args.end_time,
                                                    test_start_time=args.test_start_time) 
                
                models=[]
                # 创建模型并训练===========================================================   
                for algorithm in args.algorithms.split('+'):
                    if algorithm in ['lgb','xgb']:
                        param = {    
                            'n_estimators': 80,  # 树的数量
                            'learning_rate': 0.12,              # 学习率
                            # # 'max_depth': 3,                   # 树的最大深度    
                            # # 'subsample': 0.9,                 # 样本子采样比例   
                            # # 'colsample_bytree': 0.7,          # 特征子采样比例  
                            # # 'objective': 'reg:squarederror',  # 回归目标函数    
                            'metric': 'mse',            # 评估指标为均方根误差   
                            # # 'min_child_weight': 3,           # 增大最小叶节点权重 
                            # # 'gamma': 0.1,                     # 分裂所需的最小增益  
                            # 'reg_alpha': 0.5,                 # L1 正则化系数  
                            # 'reg_lambda': 1.0,                # L2 正则化系数
                            'verbose':-1
                            }
                    elif algorithm in ['tablenet']:
                        param={'input_dim':trainval[feas].shape[1],
                               'output_dim':16
                               }
                    else:
                        param={}
                        
                    model = Model(base_model=algorithm,param=param)
                    model.fit(trainval[feas], trainval['y'])
                    prefix=f"{id}_{algorithm}" ##前缀
                    model.save(f"model/{prefix}.pkl")
                    models.append(model)
                
                # 组合模型
                combiner_model = ModelCombiner(models=models)
                combiner_model.fit(trainval[feas], trainval['y'])
                prefix=f"{id}_combiner_{args.algorithms}" ##前缀
                combiner_model.save(f"model/{prefix}.pkl")
                #在训练验证集上的效果
                y_trainval_pred=combiner_model.predict(trainval[feas])
                trainval_res=General.get_res(y_trainval_pred, trainval['y'])     
                trainval_res[['suspendFlag','return','open','close','preclose','volume','amount']]=trainval[['suspendFlag','return','open','close','preclose','volume','amount']]
                # trainval_res=General.move_down_date(trainval_res)
                trainval_ress.append([id,trainval_res])
                
                #在测试集上的效果
                y_test_pred=combiner_model.predict(test[feas])
                test_res=General.get_res(y_test_pred, test['y']) ##单个id测试集结果   
                test_res[['suspendFlag','return','open','close','preclose','volume','amount']]=test[['suspendFlag','return','open','close','preclose','volume','amount']]
                test_ress.append([id,test_res])               
                del fe_data, y_fe_data, models, combiner_model, trainval, test
                gc.collect()
            except:
                traceback.print_exc()        
                failed_ids.append(id)   
                
        return trainval_ress, test_ress, failed_ids
    
    def predict(self,args):
        pred_ress=[]
        for id in args.ids:
            print(f'{id} ###################')
            try:
                #读取id对应配置
                config=ConfigParser(id)    
                
                fe_data=self.read_fe_data(config)
                predict=fe_data[args.start_time:]    
                
                feas=deepcopy(args.fea_select)
                
                # path='config/can_use_pe.csv' 
                # df=pd.read_csv(path)
                # can_use_pe_ids=list(df['id'])
                
                # if id not in can_use_pe_ids: ##有些票没有pe数据，就不用
                #     feas.remove('pe')     
                    
                models=[]
                # 加载模型===========================================================   
                for algorithm in args.algorithms.split('+'):
                    prefix=f"{id}_{algorithm}" ##前缀
                    model = Model(base_model=algorithm,param={})
                    model.load(f"model/{prefix}.pkl")     
                    models.append(model)
                
                # 组合模型
                combiner_model = ModelCombiner(models=models)
                prefix=f"{id}_combiner_{args.algorithms}" ##前缀
                combiner_model.load(f"model/{prefix}.pkl")     
                y_predict_pred=combiner_model.predict(predict[feas])          
                
                predict_res=pd.DataFrame(y_predict_pred,columns=['pred'],index=predict.index)
                predict_res[['suspendFlag','return','open','close','preclose','volume','amount']]=predict[['suspendFlag','return','open','close','preclose','volume','amount']]
                pred_ress.append([id,predict_res])
                del fe_data, models, combiner_model, predict
                gc.collect()
            except:
                traceback.print_exc()  
        return pred_ress