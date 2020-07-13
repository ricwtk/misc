# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 15:55:06 2020

@author: Lenovo
"""

import pandas as pd
from sklearn.model_selection import train_test_split

# import glass.csv as DataFrame
data = pd.read_csv("glass.csv", names=["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Glass type"], index_col=0)

dt = data[['Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe']]
y = data['Glass type']
x_train,x_test,y_train,y_test = train_test_split(dt,y,test_size=0.3)


