#-*- coding:utf-8 -*-
from bayesian_model import get_data

'''
数据预处理
'''
import pandas as pd

data = get_data() #获取pd.DataFrame形式的数据

#前向填充
data.fillna(method='ffill')

#后向填充
data.fillna(method='bfill')

#中值填充
df['median'] = df[feature].rolling(window).median()
df['std'] = df[feature].rolling(window).std()
df = df[(df[feature]<= df['median'] + 3*df['std']) & (df[feature]>= df['median'] - 3*df['std'])]


