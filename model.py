import os,sys,time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import traceback
import copy
import pickle
import multiprocessing as mp
import threading as td
import concurrent.futures
import random
import itertools
import math
import joblib    

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
# 训练数据
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import xgboost as xgb
from lightgbm import log_evaluation, early_stopping
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import optuna
from tools.general_func import General

callbacks = [log_evaluation(period=100), early_stopping(stopping_rounds=30)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3, 128)  # 隐藏层1
        self.fc2 = nn.Linear(128, 64)  # 隐藏层2
        self.fc3 = nn.Linear(64, 32)  # 隐藏层2
        self.fc4 = nn.Linear(32, 1)   # 输出层
        self.relu = nn.ReLU()         # 激活函数

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)

import torch
import torch.nn as nn
import torch.nn.functional as F

class TableNet(nn.Module):
    def __init__(self, input_dim, output_dim, num_layers=2):
        super(TableNet, self).__init__()
        
        # 定义 TabNetLayer 作为内部类
        class TabNetLayer(nn.Module):
            def __init__(self, input_dim, output_dim):
                super(TabNetLayer, self).__init__()
                self.fc = nn.Linear(input_dim, output_dim)
                self.bn = nn.BatchNorm1d(output_dim)
                self.attention = nn.Linear(input_dim, output_dim)

            def forward(self, x):
                # 特征变换
                x_transformed = F.relu(self.bn(self.fc(x)))
                
                # 注意力机制
                attention_weights = F.softmax(self.attention(x), dim=-1)
                x = x_transformed * attention_weights
                
                return x

        # 初始化 TableNet 层
        self.layers = nn.ModuleList([
            TabNetLayer(input_dim if i == 0 else output_dim, output_dim)
            for i in range(num_layers)
        ])
        self.fc_out = nn.Linear(output_dim, 1)  # 输出层，产生每个输入的单个输出

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.fc_out(x)  # 每个输入产生一个输出，输出形状为 (batch_size, 1)
        return x.squeeze()


class MyLGB():
    def __init__(self):
        pass
    
    def build_model(self):
        params = {
            # 'boosting_type': 'dart',
            # 'objective': 'binary',
            # 'metric': 'binary_logloss',
            # 'n_estimators':10,
            # 'num_leaves': 50,
            # 'min_child_samples': 1,
            # 'learning_rate': 0.05,
            # 'feature_fraction': 0.8,
            # 'bagging_fraction': 0.8,
            # 'bagging_freq': 5,
            # 'device': 'gpu',
            # 'random_state': 0,
            # 'bagging_fraction_seed': 0,
            # 'feature_fraction_seed': 0,
            # 'device':'cuda',
            'verbose': -1
        }
        # return xgb.XGBRegressor(**params)  
        return lgb.LGBMRegressor(**params)  
    def train(self, x_trainval, x_test, y_trainval, y_test,method='simple'):
        model = self.build_model()
        
        if method=='optuna': ##
            model=self.finetune(x_trainval, x_test, y_trainval, y_test)
        elif method=='simple': ##直接训练
            # x_trainval, y_trainval = General.add_noise_resample(x_trainval, y_trainval, x_noise_level=0.01, y_noise_level=0.01, n_samples=2)
            model.fit(x_trainval,y_trainval)
        elif method=='cross': ##交叉训练，多折结果求平均
            model=self.cross(x_trainval, x_test, y_trainval, y_test)
            
        return model
    
    def predict(self, model,x_test):
        y_pred = model.predict(x_test)
        # leaf_ids  = model.predict(x_test, pred_leaf=True)
        y_pred=pd.DataFrame(y_pred,index=x_test.index,columns=['pred']) ##put date info into datframe index
        return y_pred

    def save(self, model,model_path):
        # model_save===========================================================        
        joblib.dump(model, model_path)
        
    def load(self, model_path):
        # model load===========================================================    
        best_model = joblib.load(model_path)
        return best_model
        
    def cross(self, x_trainval, x_test, y_trainval, y_test):
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        models=[]
        
        for train_index, val_index in kf.split(x_trainval):
            model = lgb.LGBMRegressor()
            x_train, x_val = x_trainval.iloc[train_index], x_trainval.iloc[val_index]
            y_train, y_val = y_trainval.iloc[train_index], y_trainval.iloc[val_index]
            model.fit(x_train,y_train)
            models.append(model)       
        
        y_test_preds=[]
        for model in models:
            y_test_pred = model.predict(x_test)
            y_test_preds.append(y_test_pred)
            
        mse=mean_squared_error(np.mean(y_test_preds,axis=0),y_test)
        print('test mse:',mse)
        return models
        

    def finetune(self, x_trainval, x_test, y_trainval, y_test, n_trials=100):
        
        kf = KFold(n_splits=10, shuffle=True, random_state=0)
        def objective(trial):
            
            param = {
                # 'boosting_type':'gbdt',
                # 'class_weight':None, 
                # 'colsample_bytree':1.0, 
                # 'device':'cpu',
                # 'importance_type':'split', 
                # 'learning_rate':trial.suggest_float('learning_rate', 1e-5,1e-1),
                # 'max_depth':trial.suggest_int('max_depth', 2,15,step=1),
                # 'min_child_samples':91, 
                # 'min_child_weight':0.001,
                # 'min_split_gain':0.2, 
                'n_estimators':trial.suggest_int('n_estimators', 1,300,step=2),
                'n_jobs':8, 
                # 'num_leaves':trial.suggest_int('num_leaves', 10,100,step=2),
                # 'objective':None, 
                # 'random_state':0, 
                # 'reg_alpha':trial.suggest_float('reg_alpha', 0.1, 1,step=0.1),
                # 'reg_lambda':trial.suggest_float('reg_lambda', 0.1, 1,step=0.1),
                # 'silent':True, 
                # 'subsample':trial.suggest_float('subsample', 0.1, 1,step=0.1), 
                # 'subsample_for_bin':200000,
                # 'subsample_freq':0
                'verbose': -1
            }

            model = lgb.LGBMRegressor(**param)
            mses=[]
            for train_index, val_index in kf.split(x_trainval):
                x_train, x_val = x_trainval.iloc[train_index], x_trainval.iloc[val_index]
                y_train, y_val = y_trainval.iloc[train_index], y_trainval.iloc[val_index]
                model.fit(x_train,y_train)
                y_val_pred = model.predict(x_val)
                mse=mean_squared_error(y_val,y_val_pred)
                mses.append(mse)

            return np.mean(mses)
        
        study = optuna.create_study(
            direction='minimize')  # maximize the auc,minimize
        study.optimize(objective, n_trials=n_trials)
        print("Best parameters:", study.best_params)
        best_model = lgb.LGBMRegressor( **study.best_params)
        best_model.fit(x_trainval,y_trainval)
        return best_model  
    


class Model(BaseEstimator):
    def __init__(self, base_model, param):
        self.base_model = base_model.lower()
        self.param = param
        self.model = self.build_model()
        self.scaler=MinMaxScaler()
        
    def build_model(self):
        if self.base_model == 'xgb':
            return XGBRegressor(**self.param)
        elif self.base_model == 'lgb':
            return LGBMRegressor(**self.param)
        elif self.base_model == 'knn':
            return KNeighborsRegressor(**self.param)
        elif self.base_model == 'forest':
            return RandomForestRegressor(**self.param)
        elif self.base_model == 'lasso':
            return Lasso(**self.param)
        elif self.base_model == 'lr':
            return LinearRegression(**self.param)
        elif self.base_model == 'tablenet':
            return TableNet(**self.param)
        else:
            raise ValueError("Unsupported model type")

    def fit(self, x, y, epochs=1000, batch_size=32, lr=1e-3):
        if self.base_model == 'tablenet':
            
            self.scaler = MinMaxScaler()
            x_train = self.scaler.fit_transform(x)
            y_train = np.array(y)
            x_train = x_train.astype(np.float32)
            y_train = y_train.astype(np.float32)
            x_train = torch.from_numpy(x_train).to(device)
            y_train = torch.from_numpy(y_train).to(device)
            self.model.to(device)
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.MSELoss()
    
            for epoch in range(epochs):
                self.model.train()
                optimizer.zero_grad()
                y_pred = self.model(x_train)
                loss = criterion(y_pred, y_train.view(-1, 1))
                loss.backward()
                optimizer.step()
        else:
            self.model.fit(x, y)

    def predict(self, x):
        if self.base_model == 'tablenet':
            x = torch.tensor(x, dtype=torch.float32)
            self.model.eval()
            with torch.no_grad():
                return self.model(x).detach().cpu().numpy()
        else:
            return self.model.predict(x)

    def save(self, model_path):
        if self.base_model == 'tablenet':
            # 保存模型和 scaler 到一个字典
            model_info = {
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler
            }
            torch.save(model_info, model_path)
        else:
            joblib.dump(self.model, model_path)

    def load(self, model_path):
        if self.base_model == 'tablenet':
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        else:
            self.model = joblib.load(model_path)


class ModelCombiner(BaseEstimator):
    def __init__(self, models, combiner_model=LinearRegression()):
        """
        初始化模型组合器

        参数:
        - models: list, 包含已训练的基础模型列表
        - combiner_model: sklearn 风格的模型，用于组合基础模型的输出
        """
        self.models = models
        self.combiner_model = combiner_model

    def fit(self, x, y):
        """
        训练组合模型
        - x: ndarray, 输入特征
        - y: ndarray, 标签
        """
        # 用基础模型生成预测
        model_predictions = []
        for model in self.models:
            if model.base_model == 'tablenet':
                x_train = model.scaler.transform(x)
                x_train=x_train.astype(np.float32)
                x_train = torch.from_numpy(x_train).to(device)
                pred=model.predict(x_train)
            else:
                pred=model.predict(x)
                
            model_predictions.append(pred)
        stacked_features = np.column_stack(model_predictions)
        self.combiner_model.fit(stacked_features, y)
        return self

    def predict(self, x):
        """
        使用组合模型预测
        - x: ndarray, 输入特征
        """
        # 用基础模型生成预测
        model_predictions = []
        for model in self.models:
            if model.base_model == 'tablenet':
                x_train = model.scaler.transform(x)
                x_train=x_train.astype(np.float32)
                x_train = torch.from_numpy(x_train).to(device)
                pred=model.predict(x_train)
            else:
                pred=model.predict(x)
                
            model_predictions.append(pred)
        stacked_features = np.column_stack(model_predictions)
        return self.combiner_model.predict(stacked_features)
    
    def save(self,model_path):
        joblib.dump(self.combiner_model, model_path)
        
    def load(self, model_path):
        self.combiner_model = joblib.load(model_path)
