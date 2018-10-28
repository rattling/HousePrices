# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 14:52:33 2018

@author: John
"""



import pandas as pd
df = pd.DataFrame([[1,2],[3,4]], columns=list('ab'))
#col_name = 'b'
#newColName = 'b' + '^2'
#df
#df[newColName] = df['b']**2
#df["MyString"] = "blah"


num_list = list(df.select_dtypes(exclude=['object']).columns)
for col_name in num_list:
    new_col_name = col_name + '^2'
    df[new_col_name] = df[col_name] **2
    print (col_name)


