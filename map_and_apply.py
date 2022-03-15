import pandas as pd
import numpy as np

df = pd.read_csv('citibike.csv')

usertp = {'Customer':1, 'Subscriber':2}
#print(df['usertype'].map(usertp).head(5))
#print(df.apply(min))
#print(df['tripduration'].apply(lambda x: x / 60).head(5))
print(df.apply(lambda x: x['tripduration'] / 60, axis=1).head(5))

