import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm
import traceback
import logging
import warnings
warnings.filterwarnings('ignore')

from config.config import ConfigParser,stock_info_path,stock_info

import riskfolio as rp
import yfinance as yf
from timeit import default_timer as timer
from datetime import timedelta

class Riskfolio:
    def __init__(self):
        pass

    @staticmethod
    def cal_dynamic_returns(agent,ids,date,N=30,keep_last=False):
        '''
        date和N表示，使用date日前N天数据输入riskfolio来计算权重

        '''
        def func(id):
            config=ConfigParser(id)
            df=agent.read_fe_data(config)
            # df['return']=df['close']/df['preclose']-1
            return df[['return']] ##fe_data中已经包含了当日的return，注意区别于label
        riskfolio_returns=list(map(func,ids))
        riskfolio_returns=pd.concat(riskfolio_returns,axis=1)
        riskfolio_returns.columns=ids
        riskfolio_returns=riskfolio_returns.dropna()     
        riskfolio_returns=riskfolio_returns[:date]
        if keep_last:
            riskfolio_returns=riskfolio_returns.iloc[-(N):]    
        else:
            riskfolio_returns=riskfolio_returns.iloc[-(N+1):-1]        
        return riskfolio_returns
    
    @staticmethod
    def classic_mean_risk_optimization(returns,asset_classes,task,plot=True):
        '''
        Mean-Variance Portfolios和Mean-Risk Portfolios都是现代投资组合理论中的概念，用于帮助投资者在风险和回报之间找到最佳平衡。尽管它们在目标上相似，即在不同的风险水平上最大化回报，但它们在考虑风险的方式上存在差异。

        Mean-Variance Portfolios
        Mean-Variance Portfolios基于Harry Markowitz于1952年提出的现代投资组合理论，也称为均值-方差优化。这种方法的核心在于投资组合的选择不仅取决于其预期回报（均值）而且还取决于其风险（方差或标准差）。Markowitz的理论认为，通过分散投资组合，可以在不同的风险水平上最大化预期回报。在这种方法中，风险是通过投资组合回报的方差或标准差来衡量的，这反映了投资回报的波动性。
        
        Mean-Risk Portfolios
        Mean-Risk Portfolios也旨在平衡回报和风险，但它们在定义风险时可能采用不同于方差的其他风险度量。这些风险度量可以包括但不限于下行风险、Value at Risk (VaR)、Conditional Value at Risk (CVaR)、或其他风险度量标准。这种方法认识到不同的投资者可能对风险有不同的容忍度，特别是在对风险的不同方面更为敏感时（例如，更关心损失的可能性而不是收益的波动性）。
        '''
        if task=='estimating_mean_variance_portfolios':
            '''
            2.1 Calculating the portfolio that maximizes Sharpe ratio.
            '''
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)  
            # Calculating optimal portfolio      
            # Select method and estimate input parameters:      
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.        
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)   
            # Estimate optimal portfolio:   
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'MV' # Risk measure used, this time will be variance
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'     
            
            constraints=pd.DataFrame([[False,'All Assets','','','<=',0.05,'','','',''],
                             [False,'All Assets','','','>=',0.01,'','','','']],
                            columns=['Disabled','Type','Set','Position','Sign','Weight','Type Relative','Relative Set','Relative','Factor'])
            A, B = rp.assets_constraints(constraints, asset_classes)
            port.ainequality = A
            port.binequality = B
            w = port.optimization(model=model, rm=rm, kelly='approx',obj=obj, rf=rf, l=l, hist=hist)
            
            if plot:
                '''
                2.2 Plotting portfolio composition
                '''
                # Plotting the composition of the portfolio
                ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                                 height=6, width=10, ax=None)
                plt.show()            
                '''
                2.3 Calculate efficient frontier
                '''        
                points = 50 # Number of points of the frontier     
                frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)  
                # Plotting the efficient frontier       
                label = 'Max Risk Adjusted Return Portfolio' # Title of point
                mu = port.mu # Expected returns
                cov = port.cov # Covariance matrix
                returns = port.returns # Returns of the assets
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)                
                plt.show()            
                # Plotting efficient frontier composition
                ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
                plt.show()   
            return w  ##返回配置资产权重
        
        elif task=='estimating_mean_risk_portfolios':
            '''
            3.1 Calculating the portfolio that maximizes Return/CVaR ratio.
            '''  
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)  
            # Calculating optimal portfolio      
            # Select method and estimate input parameters:      
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.        
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)   
            # Estimate optimal portfolio:   
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'               
            
            rm = 'CVaR' # Risk measure
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
            if plot:
                '''
                3.2 Plotting portfolio composition
                '''            
                ax = rp.plot_pie(w=w, title='Sharpe Mean CVaR', others=0.05, nrow=25, cmap = "tab20",
                                 height=6, width=10, ax=None)
                plt.show()  
                '''
                3.3 Calculate efficient frontier
                '''                  
                points = 50 # Number of points of the frontier
                frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
                label = 'Max Risk Adjusted Return Portfolio' # Title of point    
                mu = port.mu # Expected returns
                cov = port.cov # Covariance matrix
                returns = port.returns # Returns of the assets
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)                
                plt.show() 
                # Plotting efficient frontier composition
                ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
                plt.show() 
                '''
                3.4 Calculate Optimal Portfolios for Several Risk Measures
                '''  
                # Risk Measures available:
                #
                # 'MV': Standard Deviation.
                # 'MAD': Mean Absolute Deviation.
                # 'MSV': Semi Standard Deviation.
                # 'FLPM': First Lower Partial Moment (Omega Ratio).
                # 'SLPM': Second Lower Partial Moment (Sortino Ratio).
                # 'CVaR': Conditional Value at Risk.
                # 'EVaR': Entropic Value at Risk.
                # 'WR': Worst Realization (Minimax)
                # 'MDD': Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
                # 'ADD': Average Drawdown of uncompounded cumulative returns.
                # 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
                # 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
                # 'UCI': Ulcer Index of uncompounded cumulative returns.
                
                rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
                       'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']             
                w_s = pd.DataFrame([])
                for i in rms:
                    print(i)
                    w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
                    w_s = pd.concat([w_s, w], axis=1)                
                w_s.columns = rms        
                w_s.style.format("{:.2%}").background_gradient(cmap='YlGn')
                plt.show()                 
                # Plotting a comparison of assets weights for each portfolio
                fig = plt.gcf()
                fig.set_figwidth(14)
                fig.set_figheight(6)
                ax = fig.subplots(nrows=1, ncols=1)
                w_s.plot.bar(ax=ax)
                plt.show()  
            return w  ##返回配置资产权重
        
        elif task=='constraints_on_assets_and_assets_classes':
            '''
            4.1 Creating the constraints
            '''     
            asset_classes = {'Assets': ['JCI','TGT','CMCSA','CPB','MO','APA','MMC','JPM',
                            'ZION','PSA','BAX','BMY','LUV','PCAR','TXT','TMO',
                            'DE','MSFT','HPQ','SEE','VZ','CNP','NI','T','BA'], 
                 'Industry': ['Consumer Discretionary','Consumer Discretionary',
                              'Consumer Discretionary', 'Consumer Staples',
                              'Consumer Staples','Energy','Financials',
                              'Financials','Financials','Financials',
                              'Health Care','Health Care','Industrials','Industrials',
                              'Industrials','Health Care','Industrials',
                              'Information Technology','Information Technology',
                              'Materials','Telecommunications Services','Utilities',
                              'Utilities','Telecommunications Services','Financials']}
            asset_classes = pd.DataFrame(asset_classes)
            asset_classes = asset_classes.sort_values(by=['Assets'])         
            constraints = {'Disabled': [False, False, False, False, False],
                           'Type': ['All Assets', 'Classes', 'Classes', 'Classes',
                                    'Classes'],
                           'Set': ['', 'Industry', 'Industry', 'Industry', 'Industry'],
                           'Position': ['', 'Financials', 'Utilities', 'Industrials',
                                        'Consumer Discretionary'],
                           'Sign': ['<=', '<=', '<=', '<=', '<='],
                           'Weight': [0.10, 0.2, 0.2, 0.2, 0.2],
                           'Type Relative': ['', '', '', '', ''],
                           'Relative Set': ['', '', '', '', ''],
                           'Relative': ['', '', '', '', ''],
                           'Factor': ['', '', '', '', '']}      
            constraints = pd.DataFrame(constraints)
            A, B = rp.assets_constraints(constraints, asset_classes)
            '''
            4.2 Optimize the portfolio with the constraints
            '''               
            port.ainequality = A
            port.binequality = B
            model = 'Classic'
            rm = 'MV'
            obj = 'Sharpe'
            rf = 0     
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
            if plot:
                ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                                 height=6, width=10, ax=None)            
                plt.show()              
                w_classes = pd.concat([asset_classes.set_index('Assets'), w], axis=1)
                w_classes = w_classes.groupby(['Industry']).sum()
                ax = rp.plot_pie(w=w_classes, title='Sharpe Mean Variance', others=0.05, nrow=25,
                     cmap = "tab20", height=6, width=10, ax=None)
                plt.show()  
            return w  ##返回配置资产权重        

    @staticmethod
    def mean_ulcer_index_portfolio_optimization(returns,task,plot=True):
        '''
        Ulcer Index (UI)
        Ulcer Index是一种衡量投资下跌风险或波动性的指标，特别是考虑到投资从高点下跌的幅度和持续时间。它与传统的波动性指标（如标准差）不同，因为Ulcer Index专门衡量下行风险，而不是上行和下行波动性的总和。这使其成为评估投资风险的有力工具，尤其是对于那些对下跌风险特别敏感的投资者。
        '''
        if task=='estimating_mean_ulcer_index_portfolios':
            '''
            2.1 Calculating the portfolio that maximizes Ulcer Performance Index (UPI) ratio.
            '''
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)         
            # Calculating optimal portfoli     
            # Select method and estimate input parameters:   
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.   
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)   
            # Estimate optimal portfolio:      
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'UCI' # Risk measure used, this time will be variance
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'  
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
            
            if plot:
                '''
                2.2 Plotting portfolio composition
                '''
                # Plotting the composition of the portfolio     
                ax = rp.plot_pie(w=w, title='Sharpe Mean Ulcer Index', others=0.05, nrow=25, cmap = "tab20",
                                 height=6, width=10, ax=None)
                plt.show()            
                '''
                2.3 Calculate efficient frontier
                '''        
                points = 50 # Number of points of the frontier     
                frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
                # Plotting the efficient frontier       
                label = 'Max Risk Adjusted Return Portfolio' # Title of point
                mu = port.mu # Expected returns
                cov = port.cov # Covariance matrix
                returns = port.returns # Returns of the assets        
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)             
                plt.show()            
                # Plotting efficient frontier composition
                ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
                plt.show()   
            return w  ##返回配置资产权重
        
        elif task=='estimating_risk_parity_portfolios_for_ulcer_index':
            '''
            3.1 Calculating the risk parity portfolio for Ulcer Index.
            '''  
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)         
            # Calculating optimal portfoli     
            # Select method and estimate input parameters:   
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.   
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)   
            # Estimate optimal portfolio:      
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'UCI' # Risk measure used, this time will be variance
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'  
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)            
            
            b = None # Risk contribution constraints vector         
            w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
            if plot:
                '''
                3.2 Plotting portfolio composition
                '''            
                ax = rp.plot_pie(w=w_rp, title='Risk Parity Ulcer Index', others=0.05, nrow=25, cmap = "tab20",
                                 height=6, width=10, ax=None)
                plt.show()  
                '''
                3.3 Plotting Risk Composition
                '''                  
                ax = rp.plot_risk_con(w_rp, cov=port.cov, returns=port.returns, rm=rm, rf=0, alpha=0.01,
                                      color="tab:blue", height=6, width=10, ax=None)           
                plt.show() 
          
            return w  ##返回配置资产权重
        
    @staticmethod
    def riskfolio_Lib_with_MOSEK_for_real_applications(returns,task,plot=True):
        '''
        MOSEK是一个商业数学优化软件,可以用来解决复杂的数学模型
        port.solvers = ['MOSEK'] 这里暂时不能用！！！
        '''
        if task=='estimating_max_mean_risk_portfolios_for_all_risk_measures':
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)      
            # Calculating optimum portfolio      
            # Select method and estimate input parameters:   
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.        
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)        
            # Input model parameters:           
            port.solvers = ['MOSEK'] # Setting MOSEK as the default solver
            '''
            The problem doesn't have a solution with actual input parameters
            '''
            # if you want to set some MOSEK params use this code as an example
            # import mosek
            # port.sol_params = {'MOSEK': {'mosek_params': {mosek.iparam.intpnt_solve_form: mosek.solveform.dual}}}           
            port.alpha = 0.05 # Significance level for CVaR, EVaR y CDaR 
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'
            '''
            2.1 Optimizing Process by Risk Measure
            '''
       
            # Risk Measures available:
            #
            # 'MV': Standard Deviation.
            # 'MAD': Mean Absolute Deviation.
            # 'MSV': Semi Standard Deviation.
            # 'FLPM': First Lower Partial Moment (Omega Ratio).
            # 'SLPM': Second Lower Partial Moment (Sortino Ratio).
            # 'CVaR': Conditional Value at Risk.
            # 'EVaR': Entropic Value at Risk.
            # 'WR': Worst Realization (Minimax)
            # 'MDD': Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
            # 'ADD': Average Drawdown of uncompounded cumulative returns.
            # 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
            # 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
            # 'UCI': Ulcer Index of uncompounded cumulative returns.
            
            rms = ["MV", "MAD", "MSV", "FLPM", "SLPM", "CVaR",
                   "EVaR", "WR", "MDD", "ADD", "CDaR", "UCI", "EDaR"]
            w = {}
            for rm in rms:
                start = timer()
                w[rm] = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
                end = timer()
                print(rm + ' takes ',timedelta(seconds=end-start))
            '''
            2.2 Portfolio Weights
            '''
            w_s = pd.DataFrame([])
            for rm in rms:
                w_s = pd.concat([w_s, w[rm]], axis=1)
            w_s.columns = rms    
            '''
            2.3 In sample CAGR by Portfolio
            '''     
            # a1 = datetime.strptime(data.index[0], '%d-%m-%Y')
            # a2 = datetime.strptime(data.index[-1], '%d-%m-%Y')
            a1=data.index[0]
            a2=data.index[-1]
            days = (a2-a1).days
            
            cagr = {} 
            for rm in rms:
                a = np.prod(1 + returns @ w_s[rm]) ** (360/days)-1
                cagr[rm] = [a]
            
            cagr = pd.DataFrame(cagr).T
            cagr.columns = ['CAGR']
            
            cagr.style.format("{:.2%}").background_gradient(cmap='RdYlGn')            

            return w  ##返回配置资产权重
        
        elif task=='estimating_min_risk_portfolios_for_all_risk_measures':
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)      
            # Calculating optimum portfolio      
            # Select method and estimate input parameters:   
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.        
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)        
            # Input model parameters:           
            port.solvers = ['MOSEK'] # Setting MOSEK as the default solver
            # if you want to set some MOSEK params use this code as an example
            # import mosek
            # port.sol_params = {'MOSEK': {'mosek_params': {mosek.iparam.intpnt_solve_form: mosek.solveform.dual}}}           
            port.alpha = 0.05 # Significance level for CVaR, EVaR y CDaR 
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'
            '''
            3.1 Optimizing Process by Risk Measure
            '''  
            rms = ["MV", "MAD", "MSV", "FLPM", "SLPM", "CVaR",
                   "EVaR", "WR", "MDD", "ADD", "CDaR", "UCI", "EDaR"]    
            w_min = {}
            obj = 'MinRisk'
            for rm in rms:
                start = timer()
                w_min[rm] = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
                end = timer()
                print(rm + ' takes ',timedelta(seconds=end-start))
            '''
            3.2 Portfolio Weights
            '''         
            w_min_s = pd.DataFrame([])
            for rm in rms:
                w_min_s = pd.concat([w_min_s, w_min[rm]], axis=1)    
            w_min_s.columns = rms     
            '''
            3.3 In sample CAGR by Portfolio
            '''         
            from datetime import datetime
            a1 = datetime.strptime(data.index[0], '%d-%m-%Y')
            a2 = datetime.strptime(data.index[-1], '%d-%m-%Y')
            days = (a2-a1).days      
            min_cagr = {} 
            for rm in rms:
                a = np.prod(1 + returns @ w_min_s[rm]) ** (360/days)-1
                min_cagr[rm] = [a]
            
            min_cagr = pd.DataFrame(min_cagr).T
            min_cagr.columns = ['CAGR']
            
            min_cagr.style.format("{:.2%}").background_gradient(cmap='RdYlGn')            
            
            return w  ##返回配置资产权重
        
        elif task=='estimating_risk_parity_portfolios_for_all_risk_measures':
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)      
            # Calculating optimum portfolio      
            # Select method and estimate input parameters:   
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.        
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)        
            # Input model parameters:           
            port.solvers = ['MOSEK'] # Setting MOSEK as the default solver
            # if you want to set some MOSEK params use this code as an example
            # import mosek
            # port.sol_params = {'MOSEK': {'mosek_params': {mosek.iparam.intpnt_solve_form: mosek.solveform.dual}}}           
            port.alpha = 0.05 # Significance level for CVaR, EVaR y CDaR 
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'            
            '''
            4.1 Optimizing Process by Risk Measure
            '''     
            rms = ["MV", "MAD", "MSV", "FLPM", "SLPM",
                   "CVaR", "EVaR", "CDaR", "UCI", "EDaR"]
            b = None # Risk contribution constraints vector, when None is equally risk per asset
            w_rp = {}
            for rm in rms:
                start = timer()
                w_rp[rm] = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
                end = timer()
                print(rm + ' takes ',timedelta(seconds=end-start))
            '''
            4.2 Portfolio Weights
            '''     
            w_rp_s = pd.DataFrame([])
            for rm in rms:
                w_rp_s = pd.concat([w_rp_s, w_rp[rm]], axis=1)          
            w_rp_s.columns = rms              
            '''
            4.3 In sample CAGR
            '''     
            rp_cagr = {} 
            for rm in rms:
                a = np.prod(1 + returns @ w_rp_s[rm]) ** (360/days)-1
                rp_cagr[rm] = [a]    
            rp_cagr = pd.DataFrame(rp_cagr).T
            rp_cagr.columns = ['CAGR']        
            rp_cagr.style.format("{:.2%}").background_gradient(cmap='RdYlGn')             
            return w  ##返回配置资产权重                  
        
    @staticmethod
    def logarithmic_mean_risk_optimization_kelly_criterion(returns,task,plot=True):
        '''
        对数平均风险优化凯利准则"是一种投资策略，结合了对数平均风险模型和凯利准则。这个策略旨在最大化投资组合的长期增长率，并在风险和收益之间取得平衡。具体来说：        
        对数平均风险模型（Logarithmic Mean Risk Model）：这个模型是一种衡量投资组合风险和收益的方法，考虑了不同投资之间的相关性。它与传统的均值-方差模型不同，因为它使用对数收益率而不是原始收益率，这有助于更好地处理极端风险事件。
        凯利准则（Kelly Criterion）：这是一种用来计算投注比例的公式，最初是为赌博场景设计的。它告诉你在每一次投注中应该下注多少钱，以最大化你的长期增长率。凯利准则考虑了赌注的概率和赔率，以及你的资金规模。
        '''
        if task=='estimating_logarithmic_mean_variance_portfolios':
            '''
            2.1 Calculating the portfolio that maximizes Risk Adjusted Return.
            '''
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)        
            # Calculating optimal portfolio        
            # Select method and estimate input parameters:         
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.          
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)        
            # Estimate optimal portfolio:  
            # port.solvers = ['MOSEK']
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'MV' # Risk measure used, this time will be variance
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'      
            w_1 = port.optimization(model=model, rm=rm, obj=obj, kelly=False, rf=rf, l=l, hist=hist)
            w_2 = port.optimization(model=model, rm=rm, obj=obj, kelly='approx', rf=rf, l=l, hist=hist)
            w_3 = port.optimization(model=model, rm=rm, obj=obj, kelly='exact', rf=rf, l=l, hist=hist)
            w = pd.concat([w_1, w_2, w_3], axis=1)
            w.columns = ['Arithmetic', 'Log Approx', 'Log Exact']
            
            if plot:
                fig, ax = plt.subplots(figsize=(14,6))
                w.plot(kind='bar', ax = ax)
                plt.show()            
                returns = port.returns
                cov = port.cov       
                y = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_1.to_numpy()))
                x = rp.Sharpe_Risk(w_1, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                print("Risk Adjusted Return:")
                print("Arithmetic", (y/x).item() * 12**0.5)          
                y = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_2.to_numpy()))
                x = rp.Sharpe_Risk(w_2, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                print("Log Approx", (y/x).item() * 12**0.5)     
                y = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_3.to_numpy()))
                x = rp.Sharpe_Risk(w_3, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                print("Log Exact", (y/x).item() * 12**0.5)  
                
                '''
                2.2 Calculate efficient frontier
                '''        
                points = 40 # Number of points of the frontier
                frontier = port.efficient_frontier(model=model, rm=rm, kelly="exact", points=points, rf=rf, hist=hist)
                # Plotting the efficient frontier
                label = 'Max Risk Adjusted Log Return Portfolio' # Title of point
                mu = port.mu # Expected returns
                cov = port.cov # Covariance matrix
                returns = port.returns # Returns of the assets
                fig, ax = plt.subplots(figsize=(10,6))
                rp.plot_frontier(w_frontier=frontier,
                                 mu=mu,
                                 cov=cov,
                                 returns=returns,
                                 rm=rm,
                                 kelly=True,
                                 rf=rf,
                                 alpha=0.05,
                                 cmap='viridis',
                                 w=w_3,
                                 label=label,
                                 marker='*',
                                 s=16,
                                 c='r',
                                 height=6,
                                 width=10,
                                 t_factor=12,
                                 ax=ax)
                
                y1 = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_1.to_numpy())) * 12 
                x1 = rp.Sharpe_Risk(w_1, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05) * 12**0.5
                
                y2 = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_2.to_numpy())) * 12 
                x2 = rp.Sharpe_Risk(w_2, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05) * 12**0.5
                
                ax.scatter(x=x1,
                           y=y1,
                           marker="^",
                           s=8**2,
                           c="b",
                           label="Max Risk Adjusted Arithmetic Return Portfolio")
                ax.scatter(x=x2,
                           y=y2,
                           marker="v",
                           s=8**2,
                           c="c",
                           label="Max Risk Adjusted Approx Log Return Portfolio")
                plt.legend()              
                plt.show()             
            return w  ##返回配置资产权重
        
        elif task=='estimating_logarithmic_mean_evar_portfolios':
            '''
            2.1 Calculating the portfolio that maximizes Risk Adjusted Return.
            '''  
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)        
            # Calculating optimal portfolio        
            # Select method and estimate input parameters:         
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.          
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)        
            # Estimate optimal portfolio:  
            # port.solvers = ['MOSEK']
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'                 
            rm = 'EVaR' # Risk measure
            
            print('optimizing...')
            w_1 = port.optimization(model=model, rm=rm, obj=obj, kelly=False, rf=rf, l=l, hist=hist)
            print(w_1)
            w_2 = port.optimization(model=model, rm=rm, obj=obj, kelly='approx', rf=rf, l=l, hist=hist)
            print(w_2)
            w_3 = port.optimization(model=model, rm=rm, obj=obj, kelly='exact', rf=rf, l=l, hist=hist)
            '''
            The problem doesn't have a solution with actual input parameters
            '''             
            print(w_3)
            w = pd.concat([w_1, w_2, w_3], axis=1)
            w.columns = ['Arithmetic', 'Log Approx', 'Log Exact']
            if plot:
                fig, ax = plt.subplots(figsize=(14,6))
                w.plot(kind='bar', ax = ax)
                plt.show()  
                returns = port.returns
                cov = port.cov
                
                y = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_1.to_numpy()))
                x = rp.Sharpe_Risk(w_1, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                print("Risk Adjusted Return:")
                print("Arithmetic", (y/x).item() * 12**0.5)
                
                y = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_2.to_numpy()))
                x = rp.Sharpe_Risk(w_2, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                print("Log Approx", (y/x).item() * 12**0.5)
                
                y = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_3.to_numpy()))
                x = rp.Sharpe_Risk(w_3, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                print("Log Exact", (y/x).item() * 12**0.5)
                '''
                3.2 Calculate efficient frontier
                '''                  
                points = 40 # Number of points of the frontier
                
                frontier = port.efficient_frontier(model=model, rm=rm, kelly="exact", points=points, rf=rf, hist=hist)
                # Plotting the efficient frontier
                
                label = 'Max Risk Adjusted Log Return Portfolio' # Title of point
                mu = port.mu # Expected returns
                cov = port.cov # Covariance matrix
                returns = port.returns # Returns of the assets
                
                fig, ax = plt.subplots(figsize=(10,6))
                rp.plot_frontier(w_frontier=frontier,
                                 mu=mu,
                                 cov=cov,
                                 returns=returns,
                                 rm=rm,
                                 kelly=True,
                                 rf=rf,
                                 alpha=0.05,
                                 cmap='viridis',
                                 w=w_3,
                                 label=label,
                                 marker='*',
                                 s=16,
                                 c='r',
                                 height=6,
                                 width=10,
                                 t_factor=12,
                                 ax=ax)
                
                y1 = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_1.to_numpy())) * 12
                x1 = rp.Sharpe_Risk(w_1, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05) * 12**0.5
                
                y2 = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_2.to_numpy())) * 12
                x2 = rp.Sharpe_Risk(w_2, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05) * 12**0.5
                
                ax.scatter(x=x1,
                           y=y1,
                           marker="^",
                           s=8**2,
                           c="b",
                           label="Max Risk Adjusted Arithmetic Return Portfolio")
                ax.scatter(x=x2,
                           y=y2,
                           marker="v",
                           s=8**2,
                           c="c",
                           label="Max Risk Adjusted Approx Log Return Portfolio")
                plt.legend()
                plt.show() 
            return w  ##返回配置资产权重
        
        elif task=='estimating_logarithmic_mean_edar_portfolios':
            '''
            3.1 Calculating the portfolio that maximizes Risk Adjusted Return.
            '''     
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)        
            # Calculating optimal portfolio        
            # Select method and estimate input parameters:         
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.          
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)        
            # Estimate optimal portfolio:  
            # port.solvers = ['MOSEK']
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'               
            rm = 'EDaR' # Risk measure
            print('optimizing...')
            w_1 = port.optimization(model=model, rm=rm, obj=obj, kelly=False, rf=rf, l=l, hist=hist)
            print(w_1)
            w_2 = port.optimization(model=model, rm=rm, obj=obj, kelly='approx', rf=rf, l=l, hist=hist)
            print(w_2)
            w_3 = port.optimization(model=model, rm=rm, obj=obj, kelly='exact', rf=rf, l=l, hist=hist)
            print(w_3)
            w = pd.concat([w_1, w_2, w_3], axis=1)
            w.columns = ['Arithmetic', 'Log Approx', 'Log Exact']
            
            
            if plot:
                fig, ax = plt.subplots(figsize=(14,6))
                w.plot(kind='bar', ax = ax)         
                plt.show()              
                returns = port.returns
                cov = port.cov
                
                y = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_1.to_numpy()))
                x = rp.Sharpe_Risk(w_1, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                print("Risk Adjusted Return:")
                print("Arithmetic", (y/x).item() * 12)
                
                y = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_2.to_numpy()))
                x = rp.Sharpe_Risk(w_2, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                print("Log Approx", (y/x).item() * 12)
                
                y = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_3.to_numpy()))
                x = rp.Sharpe_Risk(w_3, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                print("Log Exact", (y/x).item() * 12)
                '''
                3.2 Calculate efficient frontier
                '''                  
                # points = 40 # Number of points of the frontier

                # frontier = port.efficient_frontier(model=model, rm=rm, kelly="approx", points=points, rf=rf, hist=hist)
                # # Plotting the efficient frontier

                # label = 'Max Risk Adjusted Log Return Portfolio' # Title of point
                # mu = port.mu # Expected returns
                # cov = port.cov # Covariance matrix
                # returns = port.returns # Returns of the assets
                
                # fig, ax = plt.subplots(figsize=(10,6))
                # rp.plot_frontier(w_frontier=frontier,
                #                  mu=mu,
                #                  cov=cov,
                #                  returns=returns,
                #                  rm=rm,
                #                  kelly=True,
                #                  rf=rf,
                #                  alpha=0.05,
                #                  cmap='viridis',
                #                  w=w_3,
                #                  label=label,
                #                  marker='*',
                #                  s=16,
                #                  c='r',
                #                  height=6,
                #                  width=10,
                #                  t_factor=12,
                #                  ax=ax)
                
                # y1 = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_1.to_numpy())) * 12
                # x1 = rp.Sharpe_Risk(w_1, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                
                # y2 = 1/(returns.shape[0]) * np.sum(np.log(1 + returns @ w_2.to_numpy())) * 12
                # x2 = rp.Sharpe_Risk(w_2, cov=cov, returns=returns, rm=rm, rf=rf, alpha=0.05)
                
                # ax.scatter(x=x1,
                #            y=y1,
                #            marker="^",
                #            s=8**2,
                #            c="b",
                #            label="Max Risk Adjusted Arithmetic Return Portfolio")
                # ax.scatter(x=x2,
                #            y=y2,
                #            marker="v",
                #            s=8**2,
                #            c="c",
                #            label="Max Risk Adjusted Approx Log Return Portfolio")
                # plt.legend()
                # plt.show()  
            return w  ##返回配置资产权重        

    @staticmethod
    def comparing_covariance_estimators_methods(returns,task,plot=True):
        ''' 
        
        '''
        if task=='estimating_mean_variance_portfolios':
            '''
            2.1 Calculating the portfolio that minimizes Variance.
            '''
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)
            
            # Calculating optimal portfolio
            
            # Select method and estimate input parameters:
            
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_covs = ['hist', 'ledoit', 'oas', 'shrunk', 'gl', 'ewma1',
                           'ewma2','jlogo', 'fixed', 'spectral', 'shrink',
                           'gerber1', 'gerber2']
            
            # Estimate optimal portfolio:
            
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'MV' # Risk measure used, this time will be variance
            obj = 'MinRisk' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'
            
            w_s = pd.DataFrame([])
            
            for i in method_covs:
                print(i)
                port.assets_stats(method_mu=method_mu, method_cov=i, d=0.94)
                w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
                w_s = pd.concat([w_s, w], axis=1)
                
            w_s.columns = method_covs
            if plot:
                # Plotting a comparison of assets weights for each portfolio
                fig, ax = plt.subplots(figsize=(14,6))
                
                w_s.plot.bar(ax=ax, width=0.8)
                plt.show()            
                obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe

                w_s = pd.DataFrame([])
                
                for i in method_covs:
                    port.assets_stats(method_mu=method_mu, method_cov=i, d=0.94)
                    w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
                    w_s = pd.concat([w_s, w], axis=1)
                    
                w_s.columns = method_covs
                # Plotting a comparison of assets weights for each portfolio
                ax, fig = plt.subplots(figsize=(14,6))
                
                w_s.plot.bar(ax=fig, width=0.8)
                plt.show()   

    @staticmethod
    def owa_portfolio_optimization(returns,task,plot=True):
        ''' 
        OWA的基本概念
        OWA操作是一种集聚操作，由Yager在1988年提出，主要用于多标准决策制定。OWA操作允许决策者通过指定不同的权重来控制决策过程中的优势性（optimism）或悲观性（pessimism）水平。这些权重不是赋予决策问题中的各个准则（如投资回报率、风险等），而是赋予排序后的结果值，使得OWA操作可以根据决策者的风险偏好灵活调整决策结果。
        OWA在投资组合优化中的应用
        在投资组合优化中，OWA方法可以根据投资者的风险接受度来调整权重，通过这种方式，可以综合考虑收益最大化和风险最小化的目标。具体来说，OWA允许投资者在评估潜在投资组合时，不仅仅考虑预期收益和风险，还可以考虑到收益分布的形状，如偏度和峰度，甚至是更复杂的风险度量。
        '''
        if task=='estimating_owa_portfolios':
            '''
            2.1 Comparing Classical formulations vs OWA formulations.
            '''
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)
            
            # Calculating optimum portfolio
            
            # Select method and estimate input parameters:
            
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.
            
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
            
            # Estimate optimal portfolios:
            
            # port.solvers = ['MOSEK'] # It is recommended to use mosek when optimizing GMD
            # port.sol_params = {'MOSEK': {'mosek_params': {mosek.iparam.num_threads: 2}}}
            alpha = 0.05
            
            port.alpha = alpha
            model ='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rms = ['CVaR', 'WR'] # Risk measure used, this time will be CVaR and Worst Realization
            objs = ['MinRisk', 'Sharpe'] # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'
            
            ws = pd.DataFrame([])
            for rm in rms:
                for obj in objs:
                    print(rm,obj)
                    # Using Classical models
                    w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
                    # Using OWA model
                    if rm == "CVaR":
                        owa_w = rp.owa_cvar(len(returns), alpha=alpha)
                    elif rm == 'WR':
                        owa_w = rp.owa_wr(len(returns))
                    w1 = port.owa_optimization(obj=obj, owa_w=owa_w, rf=rf, l=l)
                    ws1 = pd.concat([w, w1], axis=1)
                    ws1.columns = ['Classic ' + obj + ' ' + rm, 'OWA ' + obj + ' ' + rm]
                    ws1['diff ' + obj + ' ' + rm] = ws1['Classic ' + obj + ' ' + rm] - ws1['OWA ' + obj + ' ' + rm]
                    ws = pd.concat([ws, ws1], axis=1)
            
            ws.style.format("{:.2%}").background_gradient(cmap='YlGn', vmin=0, vmax=1)

    @staticmethod
    def mean_semi_kurtosis_optimization(returns,task,plot=True):
        ''' 
        "Mean semi-kurtosis optimization" 是指一种金融优化策略，它考虑了资产的平均半峰值（mean semi-kurtosis）。峰度（kurtosis）是描述概率分布形状或“尾部厚度”的统计量。半峰值（semi-kurtosis）特指相对于均值的分布峰度，通常关注分布的尾部。

        在金融领域，基于平均半峰值的优化可能涉及找到一种资产配置，既能最小化半峰值所指示的下行风险，又能最大化收益。这可能是更广泛的风险管理策略的一部分，旨在减少投资组合中极端事件或异常值的影响。
        '''
        if task=='estimating_mean_semi_kurtosis_portfolios':
            '''
            2.1 Calculating the portfolio that optimize return/semi kurtosis ratio.
            '''
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)
            
            # Calculating optimum portfolio
            
            # Select method and estimate input parameters:
            
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.
            method_kurt='hist' # Method to estimate cokurtosis square matrix based on historical data.
            
            port.assets_stats(method_mu=method_mu,
                              method_cov=method_cov,
                              method_kurt=method_kurt,
                              )
            
            # Estimate optimal portfolio:
            
            # port.solvers = ['MOSEK'] # It is recommended to use mosek when optimizing GMD
            # port.sol_params = {'MOSEK': {'mosek_params': {'MSK_IPAR_NUM_THREADS': 2}}}
            
            model ='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'SKT' # Risk measure used, this time will be Tail Gini Range
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'
            
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
            
            if plot:
                '''
                2.2 Plotting portfolio composition
                '''
                # Plotting the composition of the portfolio
                
                ax = rp.plot_pie(w=w,
                                 title='Sharpe Mean - Kurtosis',
                                 others=0.05,
                                 nrow=25,
                                 cmap = "tab20",
                                 height=6,
                                 width=10,
                                 ax=None)
                plt.show()            
                '''
                2.3 Plotting risk measures
                '''        
                
                # ax = rp.plot_hist(returns=returns,
                #                   w=w,
                #                   alpha=0.05,
                #                   bins=50,
                #                   height=6,
                #                   width=10,
                #                   ax=None)          
                # plt.show()            
                points = 50 # Number of points of the frontier

                frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
                # Plotting the efficient frontier
                
                label = 'Max Risk Adjusted Return Portfolio' # Title of point
                mu = port.mu # Expected returns
                cov = port.cov # Covariance matrix
                returns = port.returns # Returns of the assets
                
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)               
                plt.show() 
                # Plotting efficient frontier composition
                ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
                plt.show()   
            return w  ##返回配置资产权重
               
        elif task=='estimating_risk_parity_portfolios_for_square_root_semi_kurtosis':
            '''
            3.1 Calculating the risk parity portfolio for Square Root Semi Kurtosis.
            '''  
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)
            
            # Calculating optimum portfolio
            
            # Select method and estimate input parameters:
            
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.
            method_kurt='hist' # Method to estimate cokurtosis square matrix based on historical data.
            
            port.assets_stats(method_mu=method_mu,
                              method_cov=method_cov,
                              method_kurt=method_kurt,
                              )
            
            # Estimate optimal portfolio:
            
            # port.solvers = ['MOSEK'] # It is recommended to use mosek when optimizing GMD
            # port.sol_params = {'MOSEK': {'mosek_params': {'MSK_IPAR_NUM_THREADS': 2}}}
            
            model ='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'SKT' # Risk measure used, this time will be Tail Gini Range
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'
            
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)   
            
            b = None # Risk contribution constraints vector

            w_rp = port.rp_optimization(model=model, rm=rm, rf=rf, b=b, hist=hist)
            if plot:
                '''
                3.2 Plotting portfolio composition
                '''            
                ax = rp.plot_pie(w=w_rp,
                                 title='Risk Parity Square Root Semi Kurtosis',
                                 others=0.05,
                                 nrow=25,
                                 cmap="tab20",
                                 height=6,
                                 width=10,
                                 ax=None)
                plt.show()  
                '''
                3.3 Plotting Risk Composition
                '''                  
                ax = rp.plot_risk_con(w_rp, cov=port.cov, returns=port.returns, rm=rm, rf=0, alpha=0.05,
                                      color="tab:blue", height=6, width=10, ax=None)            
                plt.show() 
                # Plotting the efficient frontier
                ws = pd.concat([w, w_rp],axis=1)
                ws.columns = ["Max Return/ Semi Kurtosis", "Risk Parity Semi Kurtosis"]
                
                mu = port.mu # Expected returns
                cov = port.cov # Covariance matrix
                returns = port.returns # Returns of the assets
                points = 50 # Number of points of the frontier

                frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                                      rf=rf, alpha=0.05, cmap='viridis', w=ws,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)
                plt.show() 
            return w  ##返回配置资产权重

    @staticmethod
    def black_litterman_mean_risk_optimization(returns,task,plot=True):
        ''' 
        Black-Litterman模型是一种融合了传统Markowitz均值-方差优化框架和投资者个人观点的资产配置模型。这一模型的关键在于能够让投资者将自己对市场的主观看法与基于历史数据推导出的市场均衡预期收益相结合，从而得到一个更新后的预期收益向量。
        '''
        if task=='estimating_black_litterman_portfolios':
            '''
            2.1 Calculating a reference portfolio.
            '''
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)
            
            # Calculating optimal portfolio
            
            # Select method and estimate input parameters:
            
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.
            
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
            
            # Estimate optimal portfolio:
            
            port.alpha = 0.05
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'MV' # Risk measure used, this time will be variance
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'
            
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
            
            if plot:
                # Plotting the composition of the portfolio
                
                ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                                 height=6, width=10, ax=None)
                plt.show()            
                '''
                2.2 Plotting portfolio composition
                '''        
                asset_classes = {'Assets': ['JCI','TGT','CMCSA','CPB','MO','APA','MMC','JPM',
                                            'ZION','PSA','BAX','BMY','LUV','PCAR','TXT','TMO',
                                            'DE','MSFT','HPQ','SEE','VZ','CNP','NI','T','BA'], 
                                 'Industry': ['Consumer Discretionary','Consumer Discretionary',
                                              'Consumer Discretionary', 'Consumer Staples',
                                              'Consumer Staples','Energy','Financials',
                                              'Financials','Financials','Financials',
                                              'Health Care','Health Care','Industrials','Industrials',
                                              'Industrials','Health care','Industrials',
                                              'Information Technology','Information Technology',
                                              'Materials','Telecommunications Services','Utilities',
                                              'Utilities','Telecommunications Services','Financials']}
                
                asset_classes = pd.DataFrame(asset_classes)
                asset_classes = asset_classes.sort_values(by=['Assets'])
                
                views = {'Disabled': [False, False, False],
                         'Type': ['Classes', 'Classes', 'Classes'],
                         'Set': ['Industry', 'Industry', 'Industry'],
                         'Position': ['Energy', 'Consumer Staples', 'Materials'],
                         'Sign': ['>=', '>=', '>='],
                         'Weight': [0.08, 0.1, 0.09], # Annual terms 
                         'Type Relative': ['Classes', 'Classes', 'Classes'],
                         'Relative Set': ['Industry', 'Industry', 'Industry'],
                         'Relative': ['Financials', 'Utilities', 'Industrials']}
                
                views = pd.DataFrame(views)              
                P, Q = rp.assets_views(views, asset_classes)
                # Estimate Black Litterman inputs:
                port.blacklitterman_stats(P, Q/252, rf=rf, w=w, delta=None, eq=True)          
                # Estimate optimal portfolio:       
                model='BL'# Black Litterman
                rm = 'MV' # Risk measure used, this time will be variance
                obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
                hist = False # Use historical scenarios for risk measures that depend on scenarios        
                w_bl = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
                # Plotting the composition of the portfolio
                ax = rp.plot_pie(w=w_bl, title='Sharpe Black Litterman', others=0.05, nrow=25,
                                 cmap = "tab20", height=6, width=10, ax=None)                
                plt.show()  
                ''' 
                2.3 Calculate efficient frontier
                '''
                points = 50 # Number of points of the frontier             
                frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
                
                # Plotting the efficient frontier          
                label = 'Max Risk Adjusted Return Portfolio' # Title of point
                mu = port.mu_bl # Expected returns of Black Litterman model
                cov = port.cov_bl # Covariance matrix of Black Litterman model
                returns = port.returns # Returns of the assets
                
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                                      rf=rf, alpha=0.05, cmap='viridis', w=w_bl, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)            
                plt.show() 
                # Plotting efficient frontier composition          
                ax = rp.plot_frontier_area(w_frontier=frontier, cmap="tab20", height=6, width=10, ax=None)
                plt.show()   
            return w  ##返回配置资产权重
               
        elif task=='estimating_black_litterman_mean_risk_portfolios':
            '''
            3.4 Calculate Black Litterman Portfolios for Several Risk Measures
            '''  
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)
            
            # Calculating optimal portfolio
            
            # Select method and estimate input parameters:
            
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.
            
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)
            
            # Estimate optimal portfolio:
            
            port.alpha = 0.05
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'

            # Risk Measures available:
            #
            # 'MV': Standard Deviation.
            # 'MAD': Mean Absolute Deviation.
            # 'MSV': Semi Standard Deviation.
            # 'FLPM': First Lower Partial Moment (Omega Ratio).
            # 'SLPM': Second Lower Partial Moment (Sortino Ratio).
            # 'CVaR': Conditional Value at Risk.
            # 'EVaR': Entropic Value at Risk.
            # 'WR': Worst Realization (Minimax)
            # 'MDD': Maximum Drawdown of uncompounded cumulative returns (Calmar Ratio).
            # 'ADD': Average Drawdown of uncompounded cumulative returns.
            # 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
            # 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
            # 'UCI': Ulcer Index of uncompounded cumulative returns.
            
            rms = ['MV', 'MAD', 'MSV', 'FLPM', 'SLPM', 'CVaR',
                   'EVaR', 'WR', 'MDD', 'ADD', 'CDaR', 'UCI', 'EDaR']
            
            w_s = pd.DataFrame([])
            port.alpha = 0.05
            
            for i in rms:
                print(i)
                if i == 'MV':
                    hist = False
                else:
                    hist = True
                w = port.optimization(model=model, rm=i, obj=obj, rf=rf, l=l, hist=hist)
                w_s = pd.concat([w_s, w], axis=1)
                
            w_s.columns = rms
            w_s.style.format("{:.2%}").background_gradient(cmap='YlGn')
            
            if plot:
                # Plotting a comparison of assets weights for each portfolio
                
                fig = plt.gcf()
                fig.set_figwidth(14)
                fig.set_figheight(6)
                ax = fig.subplots(nrows=1, ncols=1)
                
                w_s.plot.bar(ax=ax)
                plt.show()  
            return w  ##返回配置资产权重
 
    @staticmethod
    def constraints_on_return_and_risk_measures(returns,task,plot=True):
        ''' 
        '''
        if task=='estimating_mean_variance_portfolios':
            '''
            2.1 Calculating the portfolio that maximizes Sharpe ratio.
            '''
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)         
            # Calculating optimal portfolio         
            # Select method and estimate input parameters:          
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.      
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)       
            # Estimate optimal portfolio:        
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            rm = 'MV' # Risk measure used, this time will be variance
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'
            
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)
            
            if plot:
                '''
                2.2 Plotting portfolio composition
                '''   
                # Plotting the composition of the portfolio           
                ax = rp.plot_pie(w=w, title='Sharpe Mean Variance', others=0.05, nrow=25, cmap = "tab20",
                                 height=6, width=10, ax=None)
                plt.show()            
                '''
                2.3 Calculate Efficient Frontier
                '''        
                points = 50 # Number of points of the frontier           
                frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)          
                # Plotting the efficient frontier in Std. Dev. dimension       
                label = 'Max Risk Adjusted Return Portfolio' # Title of point
                mu = port.mu # Expected returns
                cov = port.cov # Covariance matrix
                returns = port.returns # Returns of the assets
                
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)                
                plt.show()  
                # Plotting the efficient frontier in CVaR dimension
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm='CVaR',
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)
                plt.show()  
                # Plotting the efficient frontier in Max Drawdown dimension
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm='MDD',
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)
                plt.show() 

            return w  ##返回配置资产权重
               
        elif task=='building_portfolios_with_constraints_on_return_and_risk_measures':
            # Building the portfolio object
            port = rp.Portfolio(returns=returns)         
            # Calculating optimal portfolio         
            # Select method and estimate input parameters:          
            method_mu='hist' # Method to estimate expected returns based on historical data.
            method_cov='hist' # Method to estimate covariance matrix based on historical data.      
            port.assets_stats(method_mu=method_mu, method_cov=method_cov, d=0.94)       
            # Estimate optimal portfolio:        
            model='Classic' # Could be Classic (historical), BL (Black Litterman) or FM (Factor Model)
            obj = 'Sharpe' # Objective function, could be MinRisk, MaxRet, Utility or Sharpe
            hist = True # Use historical scenarios for risk measures that depend on scenarios
            rf = 0 # Risk free rate
            l = 0 # Risk aversion factor, only useful when obj is 'Utility'    
            mu = port.mu # Expected returns
            cov = port.cov # Covariance matrix
            '''
            3.1 Estimating Risk Limits for the Available Set of Assets
            '''  
            risk = ['MV', 'CVaR', 'MDD']
            label = ['Std. Dev.', 'CVaR', 'Max Drawdown']
            alpha = 0.05
            
            for i in range(3):
                limits = port.frontier_limits(model=model, rm=risk[i], rf=rf, hist=hist)
                risk_min = rp.Sharpe_Risk(limits['w_min'], cov=cov, returns=returns, rm=risk[i], rf=rf, alpha=alpha)
                risk_max = rp.Sharpe_Risk(limits['w_max'], cov=cov, returns=returns, rm=risk[i], rf=rf, alpha=alpha)    
            
                if 'Drawdown' in label[i]:
                    factor = 1    
                else:
                    factor = 252**0.5
            
                print('\nMin Return ' + label[i] + ': ', (mu @ limits['w_min']).item() * 252)
                print('Max Return ' + label[i] + ': ',  (mu @ limits['w_max']).item() * 252)
                print('Min ' + label[i] + ': ', risk_min * factor)
                print('Max ' + label[i] + ': ', risk_max * factor)
            '''
            3.2 Calculating the portfolio that maximizes Sharpe ratio with constraints in Return, CVaR and Max Drawdown.
            '''  
            rm = 'MV' # Risk measure   
            # Constraint on minimum Return
            port.lowerret = 0.16/252  # We transform annual return to daily return          
            # Constraint on maximum CVaR
            port.upperCVaR = 0.26/252**0.5 # We transform annual CVaR to daily CVaR      
            # Constraint on maximum Max Drawdown
            port.uppermdd = 0.131  # We don't need to transform drawdowns risk measures 
            w = port.optimization(model=model, rm=rm, obj=obj, rf=rf, l=l, hist=hist)  
            if plot:
                '''
                3.3 Plotting portfolio composition
                '''  
                ax = rp.plot_pie(w=w, title='Sharpe Mean CVaR', others=0.05, nrow=25, cmap = "tab20",
                 height=6, width=10, ax=None)
                plt.show()  
                '''
                3.4 Calculate Efficient Frontier
                '''                  
                points = 50 # Number of points of the frontier
                frontier = port.efficient_frontier(model=model, rm=rm, points=points, rf=rf, hist=hist)
                # Plotting the efficient frontier in Std. Dev. dimension
                label = 'Max Risk Adjusted Return Portfolio' # Title of point            
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm=rm,
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)
                plt.show() 
                # Plotting the efficient frontier in CVaR dimension
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm='CVaR',
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)
                plt.show() 
                # Plotting the efficient frontier in Max Drawdown dimension
                ax = rp.plot_frontier(w_frontier=frontier, mu=mu, cov=cov, returns=returns, rm='MDD',
                                      rf=rf, alpha=0.05, cmap='viridis', w=w, label=label,
                                      marker='*', s=16, c='r', height=6, width=10, ax=None)
                plt.show() 
                
            return w  ##返回配置资产权重        
 
####main:
if __name__ == "__main__":
              
    # start = '2016-01-01'
    # end = '2019-12-30'
    # assets = ['JCI', 'TGT', 'CMCSA', 'CPB', 'MO', 'APA', 'MMC', 'JPM',
    #           'ZION', 'PSA', 'BAX', 'BMY', 'LUV', 'PCAR', 'TXT', 'TMO',
    #           'DE', 'MSFT', 'HPQ', 'SEE', 'VZ', 'CNP', 'NI', 'T', 'BA']
    # assets.sort()
    # # Downloading data
    # data = yf.download(assets, start = start, end = end)
    # data = data.loc[:,('Adj Close', slice(None))]
    # data.columns = assets
    # data.to_csv('riskfolio_test_data.csv')
    
    data=pd.read_csv('riskfolio_test_data.csv',index_col=['Date'],parse_dates=True)
    returns = data.pct_change().dropna()
    
    w=Riskfolio.classic_mean_risk_optimization(returns,task='estimating_mean_variance_portfolios',plot=True)
    # w=Riskfolio.classic_mean_risk_optimization(returns,task='estimating_mean_risk_portfolios',plot=True)
    # w=Riskfolio.classic_mean_risk_optimization(returns,task='estimating_mean_variance_portfolios',plot=True)
    
    # w=Riskfolio.mean_ulcer_index_portfolio_optimization(returns,task='estimating_mean_ulcer_index_portfolios',plot=True)
    # w=Riskfolio.mean_ulcer_index_portfolio_optimization(returns,task='estimating_risk_parity_portfolios_for_ulcer_index',plot=True)
    
    # w=Riskfolio.riskfolio_Lib_with_MOSEK_for_real_applications(returns,task='estimating_max_mean_risk_portfolios_for_all_risk_measures',plot=True)
    # w=Riskfolio.riskfolio_Lib_with_MOSEK_for_real_applications(returns,task='estimating_min_risk_portfolios_for_all_risk_measures',plot=True)
    
    
    # w=Riskfolio.logarithmic_mean_risk_optimization_kelly_criterion(returns,task='estimating_logarithmic_mean_variance_portfolios',plot=True)
    # w=Riskfolio.logarithmic_mean_risk_optimization_kelly_criterion(returns,task='estimating_logarithmic_mean_evar_portfolios',plot=True)
    # w=Riskfolio.logarithmic_mean_risk_optimization_kelly_criterion(returns,task='estimating_logarithmic_mean_edar_portfolios',plot=True)
    
    # w=Riskfolio.comparing_covariance_estimators_methods(returns,task='estimating_mean_variance_portfolios',plot=True)
    
    # w=Riskfolio.owa_portfolio_optimization(returns,task='estimating_owa_portfolios',plot=True)
    
    # w=Riskfolio.mean_semi_kurtosis_optimization(returns,task='estimating_mean_semi_kurtosis_portfolios',plot=True)
    # w=Riskfolio.mean_semi_kurtosis_optimization(returns,task='estimating_risk_parity_portfolios_for_square_root_semi_kurtosis',plot=True)
    
    # w=Riskfolio.black_litterman_mean_risk_optimization(returns,task='estimating_black_litterman_portfolios',plot=True)
    # w=Riskfolio.black_litterman_mean_risk_optimization(returns,task='estimating_black_litterman_mean_risk_portfolios',plot=True)
    
    # w=Riskfolio.constraints_on_return_and_risk_measures(returns,task='estimating_mean_variance_portfolios',plot=True)
    # w=Riskfolio.constraints_on_return_and_risk_measures(returns,task='building_portfolios_with_constraints_on_return_and_risk_measures',plot=True)