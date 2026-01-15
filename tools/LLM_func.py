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
import math
from datetime import datetime,timedelta
import re

import os
from volcenginesdkarkruntime import Ark


class LLM:
    def __init__(self):
        pass

    @staticmethod
    def answer(input_str):
        api_key = 'b9c61a3a-8646-434b-b80e-502525771720'
        # 从环境变量中获取您的API KEY，配置方法见：https://www.volcengine.com/docs/82379/1399008
        client = Ark(
            base_url='https://ark.cn-beijing.volces.com/api/v3',
            api_key=api_key,
        )
        
        tools = [{
            "type": "web_search",
            "max_keyword": 3,  
        }]
        
        # 创建一个对话请求
        response = client.responses.create(
            model="doubao-seed-1-6-250615",
            input=[{"role": "user", "content": input_str}],
            # tools=tools,
            thinking={
             "type": "disabled", # 不使用深度思考能力
             # "type": "enabled", # 使用深度思考能力
             # "type": "auto", # 模型自行判断是否使用深度思考能力
         },
        )
        res=response.to_dict()
        return res
