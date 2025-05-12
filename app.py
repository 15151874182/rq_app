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
from datetime import datetime, timedelta
import xlsxwriter
import argparse

import tkinter as tk
from tkinter import filedialog,ttk
import rqdatac
import talib
rqdatac.init()

# 创建主窗口
root = tk.Tk()
root.title("微盘股择时评估app")
w=1366
h=768
total_turnover=''

root.geometry(f"{w}x{h}")
root.resizable(width=False, height=False)
canvas = tk.Canvas(root, width=w, height=h)
canvas.pack()

# 在画布上画直线
line1 = canvas.create_line(w/2, h*0.6, w/2, h*0.07, fill="black", width=1)
line2 = canvas.create_line(0, h*0.07, w, h*0.07, fill="black", width=1)
line3 = canvas.create_line(0, h*0.6, w, h*0.6, fill="black", width=1)

###############################第一行
# 输入待评估日期
text1 = tk.Text(root, height=1, width=150)
text1.place(x=w*0.2,y=h*0.02,width=150,height=25)
text1.insert(tk.END, '请输入待评估日期')

# 打开历史拥挤度记录文件
def open_csv():
    global wpg_hist
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        try:
            wpg_hist = pd.read_csv(file_path)
            button2.config(text="读取文件成功")
        except Exception as e:
            print(f"读取文件时出错：{e}")
            button2.config(text="读取文件时出错")
            
button2 = tk.Button(root, text="打开历史拥挤度记录文件", command=open_csv)
button2.place(x=w*0.45,y=h*0.02,width=150,height=25)

# 计算
def cacl1():
    date = text1.get("1.0", tk.END).strip()
    r1=rqdatac.get_price(order_book_ids='000001.XSHG', 
              start_date=date, 
              end_date=date, 
              frequency='1d', 
              fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
              expect_df=True,time_slice=None)
    r2=rqdatac.get_price(order_book_ids='399106.XSHE', 
              start_date=date, 
              end_date=date, 
              frequency='1d', 
              fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
              expect_df=True,time_slice=None)
    wpg=rqdatac.get_price(order_book_ids='866006.RI', 
              start_date=date, 
              end_date=date, 
              frequency='1d', 
              fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
              expect_df=True,time_slice=None)

    wpg_turnover=round(wpg['total_turnover'].iloc[0],4)
    total_turnover=round(r1['total_turnover'].iloc[0]+r2['total_turnover'].iloc[0],4)
    crowdedness=round(wpg_turnover/total_turnover,4)    
    
    differences = np.abs(wpg_hist['crowdedness'] - crowdedness)
    min_index = differences.idxmin()
    percent = round(wpg_hist.loc[min_index, 'crowdedness_percent'],2)
    
    label3.config(text=f"两市总成交额:{total_turnover}")
    label4.config(text=f"微盘股成交额:{wpg_turnover}")
    label5.config(text=f"拥挤度:{crowdedness}")
    label6.config(text=f"拥挤度历史百分位:{percent}")
    
    df=rqdatac.get_price(order_book_ids='866006.RI', 
              start_date='20250101', 
              end_date=date, 
              frequency='1d', 
              fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
              expect_df=True,time_slice=None)
    df.index = df.index.get_level_values(1)
    df['return']=df['close']/df['prev_close']-1
    
    df['DIF'], df['DEA'], df['MACD'] = talib.MACD(df['close'], 
                                                fastperiod=12, 
                                                slowperiod=26, 
                                                signalperiod=9)
    def func1(window):
        # 判断单调性
        threshold=0.04
        buy = (window[1] > window[0]) and (window[2] > window[1]) and (abs(window[1]/window[0]-1)>threshold) and (abs(window[2]/window[1]-1)>threshold)
        sell = (window[0] > window[1]) and (window[1] > window[2]) and (abs(window[1]/window[0]-1)>threshold) and (abs(window[2]/window[1]-1)>threshold)
        return 1 if buy else (-1 if sell else 0)
    
    df['MACD_signal'] = df['MACD'].rolling(window=3, min_periods=1).apply(func1)
    def func3(window):
        return window[1]/window[0]-1      
    df['MACD_pct_change'] = df['MACD'].rolling(window=2, min_periods=1).apply(func3)
    
    df['MACD_flag'] = False ##True代表该天空仓
    # 标记区间的开始和结束
    flag = False  
    for i in range(len(df)):
        index=df.index[i]
        if df.loc[index, 'MACD_signal']==-1 and not flag:
            flag = True
        if df.loc[index, 'MACD_signal']==1 and flag:
            flag = False
        if flag:
            df.loc[index, 'MACD_flag'] = True
    df['MACD_flag']=df['MACD_flag'].shift(1)
    df=df.dropna()
    
    def func2(row):
        if row['MACD_flag']:
            return 0
        else:
            return row['return']
    df['MACD_return']=df.apply(func2, axis=1)
    
    df=df[['MACD','MACD_pct_change','MACD_signal']][-10:]    
    df[['MACD', 'MACD_pct_change']] = df[['MACD', 'MACD_pct_change']].round(2)
    
    columns = ['Index'] + list(df.columns)
    tree = ttk.Treeview(root, columns=columns, show="headings")
    tree['columns'] = columns
    for col in columns:
        tree.heading(col, text=col)
        tree.column(col, width=20)
    for index, row in df.iterrows():
        values = [str(index)] + [str(val) for val in row]
        tree.insert("", "end", values=values)
    
    tree.place(x=w*0.52,y=h*0.15,width=600)
    
    
button1 = tk.Button(root, text="点击开始计算", command=cacl1)
button1.place(x=w*0.7,y=h*0.02,width=150,height=25)

###############################第二行
# 拥挤度
label1 = tk.Label(root, text="拥挤度", borderwidth=1, relief="raised", font=("Arial", 16, "bold"))
label1.place(x=w*0.25-50,y=h*0.08,width=100,height=25)

# macd
label2 = tk.Label(root, text="macd", borderwidth=1, relief="raised", font=("Arial", 16, "bold"))
label2.place(x=w*0.75-50,y=h*0.08,width=100,height=25)

###############################第三行
# 两市总成交额
label3 = tk.Label(root, text="两市总成交额", borderwidth=0, relief="raised", font=("Arial", 12, "bold"))
label3.place(x=w*0.00,y=h*0.15,width=250)

# macd指标
  
    
    
# text2 = tk.Text(root)
# text2.insert(tk.END, df.to_csv(sep='\t', na_rep='nan'))
# text2.place(x=w*0.5,y=h*0.15,width=250)

###############################第四行
# 微盘股成交额
label4 = tk.Label(root, text="微盘股成交额", borderwidth=0, relief="raised", font=("Arial", 12, "bold"))
label4.place(x=w*0.00,y=h*0.20,width=250)

###############################第五行
# 拥挤度
label5 = tk.Label(root, text="拥挤度", borderwidth=0, relief="raised", font=("Arial", 12, "bold"))
label5.place(x=w*0.00,y=h*0.25,width=250)

###############################第六行
# 拥挤度历史百分位
label6 = tk.Label(root, text="拥挤度历史百分位", borderwidth=0, relief="raised", font=("Arial", 12, "bold"))
label6.place(x=w*0.00,y=h*0.30,width=250)




# 运行主循环

root.mainloop()