# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 17:18:51 2018

@author: John
"""

import pandas as pd
import numpy as np


train_objs_num = len(train)
dataset = pd.concat(objs=[abt, score], axis=0)
dataset_preprocessed = pd.get_dummies(dataset)
train_preprocessed = dataset_preprocessed[:train_objs_num]
test_preprocessed = dataset_preprocessed[train_objs_num:]
