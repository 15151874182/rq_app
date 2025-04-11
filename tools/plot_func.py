import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from copy import deepcopy
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import traceback
import logging
import warnings
warnings.filterwarnings('ignore')

# from mayavi import mlab
import seaborn as sns
import matplotlib.dates as mdates
import mplfinance as mpf
from matplotlib.ticker import MaxNLocator
from lightgbm import plot_tree
class Plot:
    def __init__(self):
        pass


    @staticmethod
    def  plot_res(res,filename,cols = ["gt","pred"],start_time = "2021-11-12",end_time=None,days = 30,maxmin=True):
        import matplotlib as mpl
    
        start_time = pd.to_datetime(start_time)
        
        plt.figure(figsize=(30,10))
        #解决中文或者是负号无法显示的情况
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        mpl.rcParams['axes.unicode_minus'] = False
        
        #取需要画图的时间段数据
        if end_time ==None:
            end_time = start_time + pd.Timedelta(days=days)
        else:
            end_time = pd.to_datetime(end_time)
    
        df=res[start_time:end_time]
        
        for col in cols:
            if maxmin==False:
                plt.plot(df[col],label=col,alpha=1,linewidth =1.5)
            elif maxmin==True:
                df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
                plt.plot(df[col],label=col,alpha=1,linewidth =1.5)
        
        plt.legend(loc="upper left",fontsize='x-large')
        plt.title(f"{filename}",fontsize='x-large')
        # plt.savefig(f"./result/{filename}.png",dpi=300,bbox_inches='tight',pad_inches=0.0)
        
    @staticmethod
    def  plot_res1(df):    

        # 加载数据
        # df = pd.read_csv('df.csv', index_col=0, parse_dates=True)
        df.index=pd.to_datetime(df.index)
        df.index.name = "date"
        df.columns = ['回测', '沪深300基准', '超额']
        # hs300 = pd.read_csv('data/000300.SH_fe.csv', index_col=0, parse_dates=True) ##steven想画图时候用hs300对标
        # hs300 = hs300[['open', 'high', 'low', 'close']]['2019-12-31':]
        # hs300 = hs300 / hs300['close'].iloc[0]
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(40, 12))
        sns.set_theme(style="darkgrid", palette="muted", font_scale=1.8, context="talk")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
        # 设置日期格式
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 每 2 个月显示一个刻度
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)  # 旋转日期标签
        # 绘制回测和实盘折线图
        sns.lineplot(data=df, x=df.index, y='回测', color="DodgerBlue", label="回测", ax=ax)
        
        # 绘制超额收益和填充区域
        sns.lineplot(data=df, x=df.index, y='超额', color="LimeGreen", label="超额", ax=ax)
        ax.fill_between(
            x=df.index,
            y1=df['超额'],
            y2=0,
            color="lightblue",
            alpha=0.5,
        )
        sns.lineplot(data=df, x=df.index, y='沪深300基准',color="Coral", label="沪深300基准")
        # mpf.plot(
        #     hs300,
        #     type="candle",         # 蜡烛图
        #     style="binance",       # 样式
        #     ax=ax,                 # 指定轴
        #     datetime_format="%Y-%m",  # 设置日期格式，精确到月
        #     show_nontrading=True,
        #     update_width_config=dict(candle_linewidth=2.0,  # 蜡烛影线宽度
        #                              candle_width=0.6),   
        #     volume=False,          # 不绘制成交量
        # )
        # 设置标题和标签
        ax.set_ylabel("净值")
        ax.set_title("产品业绩")
        ax.legend(loc="best")  # 自动选择最佳位置显示图例
        
        # 调整布局并显示图形
        plt.tight_layout()
        plt.show()
        
        
    @staticmethod
    def  plot_res2(df):    

        # 加载数据
        # df = pd.read_csv('df.csv', index_col=0, parse_dates=True)
        df.index=pd.to_datetime(df.index)
        df.index.name = "date"
        del df['portfolio_net']
        df=df[['M5_hedge_net']]
        df.columns = ['回测+实盘']
        # df.columns = ['回测+实盘', '沪深300基准', '超额']
        # hs300 = pd.read_csv('data/000300.SH_fe.csv', index_col=0, parse_dates=True) ##steven想画图时候用hs300对标
        # hs300 = hs300[['open', 'high', 'low', 'close']]['2019-12-31':]
        # hs300 = hs300 / hs300['close'].iloc[0]
        
        # 划分回测和实盘数据
        backtest_end_time = '2024-10-21'
        real_start_time = '2024-10-22'
        df_backtest = df[:backtest_end_time]
        df_real = df[real_start_time:]
        
        # 创建图形和轴
        fig, ax = plt.subplots(figsize=(40, 12))
        sns.set_theme(style="darkgrid", palette="muted", font_scale=1.8, context="talk")
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文支持
        # 设置日期格式
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))  # 每 2 个月显示一个刻度
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)  # 旋转日期标签
        # 绘制回测和实盘折线图
        sns.lineplot(data=df_backtest, x=df_backtest.index, y='回测+实盘', color="DodgerBlue", label="回测", ax=ax)
        sns.lineplot(data=df_real, x=df_real.index, y='回测+实盘', color="Crimson", label="实盘", ax=ax)
        ax.axvline(datetime(2024, 10, 22), color="gray", linestyle="--", linewidth=1.5, label=f"{real_start_time}实盘起始日")
        
        # 绘制超额收益和填充区域
        # sns.lineplot(data=df, x=df.index, y='超额', color="LimeGreen", label="超额", ax=ax)
        # ax.fill_between(
        #     x=df.index,
        #     y1=df['超额'],
        #     y2=0,
        #     color="lightblue",
        #     alpha=0.5,
        # )
        # sns.lineplot(data=df, x=df.index, y='沪深300基准',color="Coral", label="沪深300基准")
        
        # mpf.plot(
        #     hs300,
        #     type="candle",         # 蜡烛图
        #     style="binance",       # 样式
        #     ax=ax,                 # 指定轴
        #     datetime_format="%Y-%m",  # 设置日期格式，精确到月
        #     show_nontrading=True,
        #     update_width_config=dict(candle_linewidth=2.0,  # 蜡烛影线宽度
        #                              candle_width=0.6),   
        #     volume=False,          # 不绘制成交量
        # )
        # 设置标题和标签
        ax.set_ylabel("净值")
        ax.set_title("产品业绩")
        ax.legend(loc="best")  # 自动选择最佳位置显示图例
        
        # 调整布局并显示图形
        plt.tight_layout()
        plt.show()

    @staticmethod
    def  plot_res3(res,filename,cols = ["gt","pred"],start_time = "2021-11-12",end_time=None,days = 30,maxmin=True):
        
        ###用颜色+线条+标记多组合来区分很多line
        import matplotlib as mpl
    
        start_time = pd.to_datetime(start_time)
        
        plt.figure(figsize=(30,10))
        #解决中文或者是负号无法显示的情况
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        mpl.rcParams['axes.unicode_minus'] = False
        
        #取需要画图的时间段数据
        if end_time ==None:
            end_time = start_time + pd.Timedelta(days=days)
        else:
            end_time = pd.to_datetime(end_time)
    
        df=res[start_time:end_time]
        
        colors = ['red', 'green', 'blue', 'cyan', 'magenta',   
                  'black', 'orange', 'purple', 'pink']  
        line_styles = ['-', '--',]  
        markers = ['o', 's', '^', 'D', 'x']  
        for i,col in enumerate(cols):
            color = colors[i % len(colors)]
            linestyle = line_styles[i % len(line_styles)]
            marker = markers[i % len(markers)]
            if maxmin==False:
                plt.plot(df[col],label=col,alpha=1,linewidth =1.5, color=color, linestyle=linestyle, marker=marker)
            elif maxmin==True:
                df[col]=(df[col]-df[col].min())/(df[col].max()-df[col].min())
                plt.plot(df[col],label=col,alpha=1,linewidth =1.5, color=color, linestyle=linestyle, marker=marker)
        
        plt.legend(loc="upper left",fontsize='x-large')
        plt.title(f"{filename}",fontsize='x-large')
        plt.legend(
            loc='upper right',
            bbox_to_anchor=(1.08, 1),  # 微调图例位置（向右移动2%）
            borderaxespad=0.2         # 图例与轴的间距
        )
        # plt.savefig(f"./result/{filename}.png",dpi=300,bbox_inches='tight',pad_inches=0.0)        

    @staticmethod
    def  plot_stockNum_returnAnnual(stockNum,returnAnnual):
        #解决中文或者是负号无法显示的情况
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        mpl.rcParams['axes.unicode_minus'] = False
        
        plt.plot(stockNum, returnAnnual)
        plt.title(f"top股票数k-年化收益率 with N={max(stockNum)}",fontsize='x-large')
        plt.xticks(rotation=90)  # 将横轴时间标签旋转90度
        max_y=max(returnAnnual)
        max_index = np.argmax(returnAnnual)
        plt.axhline(y=max_y, color='r', linestyle='--')
        plt.text(stockNum[max_index], max_y, f'[{stockNum[max_index]},{max_y:.3f}]', ha='left')
        plt.show()      
    
    @staticmethod
    def  plot_stockNum_fees(stockNum,fees):
        #解决中文或者是负号无法显示的情况
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        mpl.rcParams['axes.unicode_minus'] = False
        
        plt.plot(stockNum, fees)
        plt.title(f"top股票数k-手续费 with N={max(stockNum)}",fontsize='x-large')
        # plt.xticks(rotation=90)  # 将横轴时间标签旋转90度
        plt.show()    
    
    @staticmethod
    def  plot_stockNum_alphas(stockNum,alphas):
        #解决中文或者是负号无法显示的情况
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        mpl.rcParams['axes.unicode_minus'] = False
        
        plt.plot(stockNum, alphas)
        plt.title(f"top股票数k-alphas with N={max(stockNum)}",fontsize='x-large')
        # plt.xticks(rotation=90)  # 将横轴时间标签旋转90度
        plt.show() 
    
    @staticmethod
    def  plot_twinx(df):
        # 创建figure和axes对象
        fig, ax1 = plt.subplots()
        
        # 绘制左边的图
        color = 'tab:blue'
        ax1.set_xlabel('时间')
        ax1.set_ylabel('skew', color=color)
        ax1.plot(df['iskew'], color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # 创建右边的y轴
        ax2 = ax1.twinx()  
        color = 'tab:red'
        ax2.set_ylabel('IC00', color=color)  
        ax2.plot(df['close'], color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        # 添加标题
        plt.title('IC00 VS skew')
        fig.autofmt_xdate(rotation=90)
        # 显示图形
        plt.show()         
    
    
    @staticmethod
    def  plot_scattor(df,col1='iskew',col2='iskew_2nd_derivative',threshold=None,day=1):       
        #解决中文或者是负号无法显示的情况
        mpl.rcParams["font.sans-serif"] = ["SimHei"]
        mpl.rcParams['axes.unicode_minus'] = False        
        # 根据label的不同值绘制散点图
        colors = {-1: 'g', 0: 'b', 1: 'r'}
        dic={1:'涨',0:'平',-1:'跌'}
        for label, color in colors.items():
            subset = df[df['class'] == label]
            # plt.scatter(subset['iskew'], subset['iskew_1st_derivative'], c=color, label=label)
            plt.scatter(subset[col1], subset[col2], c=color, label=dic[label])
        
        plt.xlabel(col1)
        plt.ylabel(col2)
        plt.legend()
        plt.title(f'{col1} vs {col2} period={day} threshold={threshold}')
        plt.show()
        
    @staticmethod
    def  plot_scattor_3d(df):       
        # 创建3D图形对象
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 定义颜色映射
        colors = {-1: 'g', 0: 'b', 1: 'r'}
        for label, color in colors.items():
            subset = df[df['class'] == label]
            ax.scatter(subset['iskew'], subset['iskew_1st_derivative'], subset['iskew_2nd_derivative'], c=color, label=label)
        
        ax.set_xlabel('iskew')
        ax.set_ylabel('iskew_1st_derivative')
        ax.set_zlabel('iskew_2nd_derivative')
        
        plt.legend()
        plt.title('3D Scatter Plot of iskew, iskew_1st_derivative, and iskew_2nd_derivative')
        plt.show() 
        
        
    
    @staticmethod
    def  plot_tree(model,tree_index): ##graphviz 树对的可视化
        plot_tree(model, tree_index=tree_index,figsize=(40, 30))
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        