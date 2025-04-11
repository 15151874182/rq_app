import os,sys,time
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import traceback
from copy import deepcopy 
import gc
from datetime import datetime, timedelta

import rqdatac
from tools.convert_func import Convert  
from tools.metrics_func import Metrics
from tools.general_func import General
from tools.plot_func import Plot
rqdatac.init()

# et=rqdatac.get_latest_trading_date()
et='2025-03-26'
st=rqdatac.get_previous_trading_date(et,n=4,market='cn')

cols_risk_factor=[
    # 市场风险因子
    'beta', 
    'residual_volatility',

    # 价值/基本面因子
    'book_to_price', 
    'dividend_yield', 
    'earnings_yield',

    # 质量/盈利因子
    'earnings_quality', 
    'profitability', 
    'earnings_variability',

    # 增长/投资因子
    'growth', 
    'investment_quality',

    # 动量与反转因子
    'momentum', 
    'longterm_reversal',

    # 流动性/规模因子
    'size', 
    'mid_cap', 
    'liquidity',

    # 杠杆因子
    'leverage'
]

cols_industry_factor=[
    # 顺周期行业（经济敏感型）
    '煤炭', '石油石化',         # 上游资源
    '有色金属', '钢铁',         # 金属材料
    '基础化工', '建材',         # 中游原材料
    '机械', '建筑',            # 基建与设备制造
    '房地产', '汽车', '家电',   # 下游消费与地产

    # 高股息/防御型行业（弱周期）
    '银行', '非银行金融',       # 金融
    '电力及公用事业', '交通运输', # 公共事业

    # 消费类行业（需求稳定型）
    '食品饮料', '农林牧渔', '医药',        # 必需消费
    '商贸零售', '消费者服务', '纺织服装', '轻工制造',  # 可选消费

    # 科技成长类行业（创新驱动型）
    '计算机', '通信', '电子', '电力设备及新能源', '传媒',

    # 其他特殊类别
    '国防军工',               # 政策驱动型
    '综合', '综合金融'         # 多元化业务
]

implicit=rqdatac.get_factor_return(st, et, 
                  factors= None, universe='whole_market',
                  method='implicit',industry_mapping='citics_2019', model = 'v2')
explicit=rqdatac.get_factor_return(st, et, 
                  factors= None, universe='whole_market',
                  method='explicit',industry_mapping='citics_2019', model = 'v2')

##要用显式因子收益率（多空组合算出来的）+行业因子收益率（显式没有这个），拼起来
factor_return=pd.concat([explicit[cols_risk_factor],implicit[cols_industry_factor]],axis=1)

##对因子按天远近加权
daily_weights=General.sum_normalize([i for i in range(1,len(factor_return)+1)])
daily_weights = pd.Series(daily_weights, index=factor_return.index)
factor_return_weighted = factor_return.multiply(daily_weights, axis=0)


########factor_return_map画图
# factor_return=pd.concat([factor_return[cols_risk_factor],
#                         factor_return[cols_industry_factor]],axis=1)

# # 创建一个 3 行 2 列的画布
# fig, axes = plt.subplots(3, 2, figsize=(30, 30))
# #解决中文或者是负号无法显示的情况
# mpl.rcParams["font.sans-serif"] = ["SimHei"]
# mpl.rcParams['axes.unicode_minus'] = False
# plt.rcParams['figure.dpi'] = 300
# plt.tight_layout(
#     pad=10.0,        # 主画布与子图之间的边距
#     w_pad=10.0,      # 子图之间的水平间距
#     h_pad=10.0       # 子图之间的垂直间距
# )

# for id,date in enumerate(factor_return.index):
#     daily_data=factor_return.loc[date]
#     risk_part=abs(daily_data[cols_risk_factor]).rank()
#     industry_part=daily_data[cols_industry_factor].rank()
#     daily_data = pd.concat([risk_part, industry_part])
    
#     factor_return_map = []
#     for risk in cols_risk_factor:
#         for industry in cols_industry_factor:
#             factor_return_map.append(abs(daily_data[risk]) + daily_data[industry])
            
#     factor_return_map=General.normalize_list(factor_return_map, lower_bound=0, upper_bound=100)
    
#     factor_return_map=np.array(factor_return_map).reshape(len(cols_risk_factor),len(cols_industry_factor))
#     factor_return_map=pd.DataFrame(factor_return_map,index=cols_risk_factor,columns=cols_industry_factor)
#     factor_return_map.index.name=''
#     factor_return_map.columns.name=''
#     x,y=divmod(id,2)
#     sns.heatmap(factor_return_map, annot=False, cmap='coolwarm', ax=axes[x, y])
#     axes[x, y].set_title(f'{date}')


########factor_return cumsum diagram
# factor_return=factor_return.cumsum()
# cols=factor_return.columns
# Plot.plot_res3(factor_return,'',cols = cols,start_time = factor_return.index[0],
#                                 end_time=factor_return.index[-1],
#                                 days = None,
#                                 maxmin=False)

########factor_return sharp 筛选策略
res=factor_return_weighted.describe()
res=res.T
res['sharp']=res['mean']/res['std'] ##计算因子收益率sharp
res['abs_sharp']=abs(res['sharp']) ##非常负的风险因子收益率也是一种市场风格偏向，要看绝对值
risk_part,industry_part=General.split_dataframe_by_index(res)


risk_part=risk_part.sort_values(['abs_sharp'],ascending=False)
risk_part=risk_part[risk_part['abs_sharp']>0.5] 
industry_part=industry_part[industry_part['sharp']>0] 
industry_part=industry_part.sort_values(['sharp'],ascending=False)


factor_return_array = []
for risk in risk_part.index:
    for industry in industry_part.index:
        x1,y1=risk_part['sharp'][risk],risk_part['abs_sharp'][risk]
        x2=industry_part['sharp'][industry]
        factor_return_array.append([risk,industry,x1,y1,x2,y1+x2])
factor_return_array=pd.DataFrame(factor_return_array,columns = ['risk', 'industry', 'risk_sharp', 'risk_abs_sharp', 'industry_sharp', 'sum_sharp'])
risk_industry_sharp=factor_return_array.sort_values(['sum_sharp'],ascending=False)

stock_industry_dict={}
stock_pool_list=[]
for industry in industry_part.index:
    stocks=rqdatac.get_industry(industry=industry, source='citics_2019', date=None, market='cn')
    stock_pool_list+=stocks
    for stock in stocks:
        stock_industry_dict[stock]=industry
exposures=rqdatac.get_factor_exposure(stock_pool_list,st,et,factors=None,industry_mapping='citics_2019', model = 'v2')

group = exposures.groupby(level=1)
stocks_score=[]
for id,item in group:
    item=item[risk_part.index]
    item_weighted = item.multiply(daily_weights, axis=0)
    exposure=item_weighted.sum()
    industry=stock_industry_dict[id] ##查找该stock对应industry
    info=risk_industry_sharp[risk_industry_sharp['industry']==industry]
    info['exposure']=list(exposure)
    stock_score=sum(info['risk_sharp']*info['exposure']+info['industry_sharp']*abs(info['exposure']))
    stocks_score.append([id,stock_score])
stocks_score=pd.DataFrame(stocks_score,columns=['id','score'])
stocks_score=stocks_score.sort_values(['score'],ascending=False)
stocks_score['name']=stocks_score['id'].apply(lambda id:rqdatac.instruments(id, market='cn').symbol)
stocks_score=stocks_score.reset_index()
stocks_score=stocks_score[~stocks_score['name'].str.contains('ST')] ##去掉st的
stocks_score=stocks_score[stocks_score['score']>0] ##去掉负分的

k=400
date=et
select=stocks_score.iloc[:k]
select['买卖价格']=select['id'].apply(lambda id:rqdatac.get_price(order_book_ids=id, 
          start_date=date, 
          end_date=date, 
          frequency='1d', 
          fields=None, adjust_type='pre', skip_suspended =False, market='cn', 
          expect_df=True,time_slice=None)['close'].iloc[0])
money=1000e4
each=money//len(select)
select['买卖数量']=select['买卖价格'].apply(lambda price:int(each//(price*100)*100))
select['买卖日期']=date
select['买卖方向']='买入'
select['证券代码']=list(select['id'].apply(lambda id:rqdatac.id_convert(id,to='normal')))

res=select[['买卖日期','证券代码', '买卖数量', '买卖价格', '买卖方向']]

acc='acc1'  ##文件名和账户名有关联
now = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")##文件名和时间有关联
path=f'./trade_log/{acc}_{now}.xlsx'  
with pd.ExcelWriter(f'{path}', engine='xlsxwriter') as writer:
    res.to_excel(writer, sheet_name='导入数据区', index=False)      
    res3=select[['id','name']]
    res3.to_excel(writer, sheet_name='股票名清单', index=False)      




