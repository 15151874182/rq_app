import rqdatac
import pandas as pd
import matplotlib.pyplot as plt
from tools.plot_func import Plot
rqdatac.init()
xx=1
# res=pd.DataFrame()
# ids=rqdatac.index_components(order_book_id='866006.RI', 
#                              date=None, 
#                              start_date='2024-3-14', 
#                              end_date='2025-3-14', 
#                              market='cn',
#                              return_create_tm=False)

# # ids=list(ids.values())[0]
# # res['id']=ids
# # res['last_price']=res['id'].apply(lambda id:rqdatac.current_snapshot(order_book_ids=id, 
# #                              market='cn').last)
# # res['name']=res['id'].apply(lambda id:rqdatac.instruments(id, market='cn').symbol)


# daily_ids=list(ids.values())
# res2=[]
# for i in range(1,len(daily_ids)):
#     res2.append(len(set(daily_ids[i])-set(daily_ids[i-1])))
    
# res=rqdatac.index_weights(order_book_id='866006.RI', date=None)

# x2=rqdatac.get_industry_mapping(source='citics_2019', date=None, market='cn')
# x2=rqdatac.get_industry_mapping(source='sws_2021', date=None, market='cn')
# # xx=rqdatac.get_industry_change(industry='123030', source='citics', level=None, market='cn')

# df=rqdatac.all_instruments(type='INDX', market='cn', date=None)
# res = df[df['symbol'].str.contains('中信')]
# res=res[res['status']=='Active']

# bk1s=set(x2['first_industry_name'])
# bk2s=set(x2['second_industry_name'])
# bk3s=set(x2['third_industry_name'])

# wrong=[]
# for bk3 in bk3s:
#     xx = res[res['symbol'].str.contains(bk3)]
#     if len(xx)==0:
#         # wrong.append(len(xx))
#         wrong.append(bk3)

# xx=rqdatac.index_indicator(['399986.XSHE'],start_date=20000101,end_date=20250317)
# xx=xx.reset_index()
# xx=xx.set_index('trade_date')
# plt.plot(xx['pe_ttm'])

# xx=rqdatac.index_components(order_book_id='CI005302.INDX', 
#                             date=None, 
#                             start_date='2024-3-14', 
#                             end_date='2025-3-14', 
#                             market='cn',
#                             return_create_tm=False)

# xx=rqdatac.get_industry(industry='123030', source='citics_2019', date=None, market='cn')

xx=rqdatac.get_price(order_book_ids='101010', 
          start_date='2013-01-04', 
          end_date='2014-01-04', 
          frequency='1d', 
          fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
          expect_df=True,time_slice=None)

# xx=rqdatac.get_factor_exposure('600519.XSHG',
#                                 '20160301','20250317',
#                                 factors=None,industry_mapping='sws_2021' )
# xx=xx.reset_index()
# xx=xx.set_index('date')
# plt.plot(xx['size'])

# xx=rqdatac.get_stock_beta('688701.XSHG', 
#                           '20230310', 
#                           '20250315', 
#                           benchmark='000300.XSHG', model = 'v1')

# xx=rqdatac.get_factor_return('20250301', '20250317', 
#                   factors= None, universe='whole_market',
#                   method='implicit',industry_mapping='citics_2019', model = 'v2')
# xx=xx.cumsum()
# # cols=xx.columns
# cols=xx.columns[:16]
# # cols=xx.columns[17:]
# Plot.plot_res(xx,'',cols = cols,start_time = xx.index[0],
#                                 end_time=xx.index[-1],
#                                 days = None,
#                                 maxmin=False)

# xx=rqdatac.econ.get_reserve_ratio(reserve_type='major',
#                                   start_date='20170101',end_date='20250318')

# xx=rqdatac.econ.get_money_supply(start_date='20170101',end_date='20250318')

# xx=xx.reset_index()
# xx=xx.sort_values(['effective_date'],ascending=True)
# xx=xx.set_index('effective_date')
# # cols=xx.columns[17:]
# # cols=['m2', 'm1','m0']
# cols=['m2_growth_yoy', 'm1_growth_yoy','m0_growth_yoy']
# Plot.plot_res(xx,'',cols = cols,start_time = xx.index[0],
#                                 end_time=xx.index[-1],
#                                 days = None,
#                                 maxmin=False)

# xx=rqdatac.econ.get_factors(factors='工业品出厂价格指数PPI_当月同比_(上年同月=100)', 
#                              start_date='20170801', end_date='20250801')

# xx=rqdatac.all_consensus_industries()
# x2=rqdatac.get_consensus_industry_rating(industries='35',start_date='20170801', end_date='20250801')

# x2=rqdatac.news.get_stock_news('000668.XSHE','2021-03-01','2022-03-01')

# x2=rqdatac.esg.get_rating('000668.XSHE')


