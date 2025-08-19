# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 14:04:26 2025

@author: 陈天宇
"""
##res 从米筐回测结果中来

benchmark_portfolio=res['sys_analyser']['benchmark_portfolio']
portfolio=res['sys_analyser']['portfolio']
xx=pd.concat([benchmark_portfolio['unit_net_value'],portfolio['unit_net_value']],axis=1)
xx.columns=['benchmark','portfolio']
xx.to_csv('zz1000_neutro.csv')

xx['benchmark_return']=xx['benchmark']/xx['benchmark'].shift(1)-1
xx['portfolio_return']=xx['portfolio']/xx['portfolio'].shift(1)-1
xx=xx.fillna(0)

xx['return']=xx['portfolio_return']-xx['benchmark_return']

xx['net']=list(Convert.returns_to_net(xx['return']))

Metrics.print_metrics(xx['return']*0.5,xx.index,0.017) 


cc=xx['2024-03-01':'2025-08-14']
xx['net'].plot()



