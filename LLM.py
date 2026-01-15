# -*- coding: utf-8 -*-
"""
Created on Mon Jan 12 13:58:11 2026

@author: 陈天宇
"""

api_key = 'b9c61a3a-8646-434b-b80e-502525771720'
import os
from volcenginesdkarkruntime import Ark

# 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
client = Ark(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=api_key,
)

tools = [{
    "type": "web_search",
    "max_keyword": 10,  
}]

# 创建一个对话请求
response = client.responses.create(
    model="doubao-seed-1-6-250615",
    input=[{"role": "user", "content": "站在2025-8月1日，只允许使用该日之前的信息，严禁使用未来数据，列出A股中你觉得未来最有上涨潜力的5个板块/题材,从当前宏观环境、政策、资金流入、题材热点等多维度评估,请直接给我答案"}],
    # tools=tools,
    thinking={
     "type": "disabled", # 不使用深度思考能力
     # "type": "enabled", # 使用深度思考能力
     # "type": "auto", # 模型自行判断是否使用深度思考能力
 },
)

print(response)
res=response.to_dict()