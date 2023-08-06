# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 19:06:16 2023

@author: Tanisha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df1 = pd.read_csv('preprocessed_n.csv', index_col = 'created_at', parse_dates = True)
df2=pd.read_csv('preprocessed_p.csv', index_col = 'created_at', parse_dates = True)
df3=pd.read_csv('preprocessed_k.csv', index_col = 'created_at', parse_dates = True)
df4=pd.read_csv('preprocessed_ec.csv', index_col = 'created_at', parse_dates = True)
df5=pd.read_csv('preprocessed_temp.csv', index_col = 'created_at', parse_dates = True)
df6=pd.read_csv('preprocessed_rel_hum.csv', index_col = 'created_at', parse_dates = True)
df7=pd.read_csv('preprocessed_pH.csv', index_col = 'created_at', parse_dates = True)
df2=df2.resample('W').mean()

df = pd.merge(df1, df2, left_index=True, right_index=True, how='inner')
df = pd.merge(df, df3, left_index=True, right_index=True, how='inner')
df = pd.merge(df, df4, left_index=True, right_index=True, how='inner')
df = pd.merge(df, df5, left_index=True, right_index=True, how='inner')
df = pd.merge(df, df6, left_index=True, right_index=True, how='inner')
df = pd.merge(df, df7, left_index=True, right_index=True, how='inner')

df2.plot()
df2.to_csv('preprocessed_p.csv', index=True)

df2=df2.resample('H').mean()
df2=df2['2022-12-01 00:00:00+05:30':'2023-01-31 23:00:00+05:30']
df2.plot()