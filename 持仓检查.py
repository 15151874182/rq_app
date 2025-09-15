# -*- coding: utf-8 -*-
"""
Created on Wed Sep 10 15:46:41 2025

@author: 陈天宇
"""
import pandas as pd

ATX_pos_file='ATX_csv/百里挑一持仓查询_20250910152604.xlsx'
df_now=pd.read_excel(ATX_pos_file,dtype={'证券代码': str})

df_target=pd.read_csv('ATX_csv/ATX_stock_2025-09_09百里挑一信用.csv')
df_target['证券代码']=df_target['证券代码'].apply(lambda x:x.split('.')[0])
df_now=df_now[['证券代码','证券名称','持仓数量']]
df_target=df_target[['证券代码','任务数量']]

res=pd.merge(df_now,df_target,how='outer',on='证券代码')
xx=res[res['持仓数量']!=res['任务数量']]
xx=1
# for id in df2['证券代码']:
#     num1=df2[df2['证券代码']==id]['任务数量'].iloc[0]
#     id=id.split('.')[0]
#     try:
#         num2=df1[df1['证券代码']==id]['持仓数量'].iloc[0]
#     except:
#         print(id)
#     if num1!=num2:
#         print(id)