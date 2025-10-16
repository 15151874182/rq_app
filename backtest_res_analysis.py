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



xx['port-index_return']=xx['portfolio_return']*0.5-xx['benchmark_return']*0.5
xx['port-index_net']=list(Convert.returns_to_net(xx['port-index_return']))

Metrics.print_metrics(xx['return'],xx.index,0.017) 


cc=xx['2024-03-01':'2025-08-14']
xx['net'].plot()


##future01
zz1000=rqdatac.futures.get_dominant_price(underlying_symbols='IM'
                                         ,start_date=20230901,end_date=20250915,
                                         frequency='1d',fields=None,
                                         adjust_type='none', 
                                         rule=0,
                                         rank=1,
                                         adjust_method='prev_close_ratio')

zz1000.index = zz1000.index.get_level_values(1)

zz1000['future01_return']=zz1000['close']/zz1000['prev_settlement']-1
xx['future01_return']=zz1000['future01_return']
xx.loc['2023-09-01','future01_return']=0
xx['port-future01_return']=xx['portfolio_return']*0.5-xx['future01_return']*0.5
# xx['port-future01_return']=xx['portfolio_return']*0.545-xx['future01_return']*0.454
xx['port-future01_net']=list(Convert.returns_to_net(xx['port-future01_return']))


##future23 目前最佳
zz1000=rqdatac.futures.get_dominant_price(underlying_symbols='IM'
                                         ,start_date=20230901,end_date=20250915,
                                         frequency='1d',fields=None,
                                         adjust_type='none', 
                                         rule=2,
                                         rank=3,
                                         adjust_method='prev_close_ratio')

zz1000.index = zz1000.index.get_level_values(1)

zz1000['future23_return']=zz1000['close']/zz1000['prev_settlement']-1
xx['future23_return']=zz1000['future23_return']
xx.loc['2023-09-01','future23_return']=0
# xx['port-future23_return']=xx['portfolio_return']*0.5-xx['future23_return']*0.5
xx['port-future23_return']=xx['portfolio_return']*0.545-xx['future23_return']*0.454
xx['port-future23_net']=list(Convert.returns_to_net(xx['port-future23_return']))

Metrics.print_metrics(xx['port-future23_return'],xx.index,0.017)

# down=xx[xx['return']<0]
# np.std(down2['return'])*np.sqrt(252)

future23_std=xx['future23_return'].std()
portfolio_std=xx['portfolio_return'].std()
corr=xx['future23_return'].corr(xx['portfolio_return'])
hedge_ratio=corr*portfolio_std/future23_std



##############################用于计算实盘中的IM beta
benchmark_portfolio=res['sys_analyser']['benchmark_portfolio']
portfolio=res['sys_analyser']['portfolio']
xx=pd.concat([benchmark_portfolio['unit_net_value'],portfolio['unit_net_value']],axis=1)
xx.columns=['benchmark','portfolio']

xx['benchmark_return']=xx['benchmark']/xx['benchmark'].shift(1)-1
xx['portfolio_return']=xx['portfolio']/xx['portfolio'].shift(1)-1
xx=xx.fillna(0)

zz1000=rqdatac.futures.get_dominant_price(underlying_symbols='IM'
                                         ,start_date=20230901,end_date=20250930,##时间要改
                                         frequency='1d',fields=None,
                                         adjust_type='none', 
                                         rule=2,
                                         rank=3,
                                         adjust_method='prev_close_ratio')

zz1000.index = zz1000.index.get_level_values(1)

zz1000['future23_return']=zz1000['close']/zz1000['prev_settlement']-1
xx['future23_return']=zz1000['future23_return']
xx.loc['2025-09-01','future23_return']=0   ##时间要改

future23_std=xx['future23_return'].std()
portfolio_std=xx['portfolio_return'].std()
corr=xx['future23_return'].corr(xx['portfolio_return'])
hedge_ratio=corr*portfolio_std/future23_std

future23_market=zz1000['settlement'].iloc[-1]*200
stock_market=1/hedge_ratio*future23_market
print('future23_market:',future23_market)
print('stock_market:',stock_market)
