# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 16:34:26 2026

@author: 陈天宇
"""

import numpy as np
from scipy.stats import norm

# 参数设置
S = 4737
K = 5000
t = 71/365
r = 0.0188
sigma = 0.197

# 计算d1和d2
d1 = (np.log(S/K) + (r + 0.5*sigma**2)*t) / (sigma*np.sqrt(t))
d2 = d1 - sigma*np.sqrt(t)

# 计算期权价格
C = S*norm.cdf(d1) - K*np.exp(-r*t)*norm.cdf(d2)

# 计算希腊字母
delta = norm.cdf(d1)
gamma = norm.pdf(d1)/(S*sigma*np.sqrt(t))
theta = -(S*norm.pdf(d1)*sigma)/(2*np.sqrt(t)) - r*K*np.exp(-r*t)*norm.cdf(d2)
vega = S*np.sqrt(t)*norm.pdf(d1)
rho = K*t*np.exp(-r*t)*norm.cdf(d2)

print(f"期权价格: {C:.2f}")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.4f}")
print(f"Theta: {theta:.2f}")
print(f"Vega: {vega:.2f}")
print(f"Rho: {rho:.2f}")