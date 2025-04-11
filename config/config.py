# -*- coding: utf-8 -*-
"""
@author: cty
"""

import os
import sys
import json
import pandas as pd


# 添加项目路径=============================================================================
project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,project_dir) 
# 读取CS.csv=============================================================================
CS_path=os.path.join('config','CS.csv')
CS = pd.read_csv(os.path.join(project_dir,CS_path),dtype={'order_book_id':str},encoding='utf-8')
INDX_path=os.path.join('config','INDX.csv')
INDX = pd.read_csv(os.path.join(project_dir,INDX_path),dtype={'order_book_id':str},encoding='utf-8')

        
