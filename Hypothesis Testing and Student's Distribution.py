import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

taxi_mex = pd.read_csv('mex_clean.csv')
taxi_bog = pd.read_csv('Data/bog_clean.csv')

sample = taxi_mex['wait_sec'].sample(n=3000)/60
# print(stats.stats.ttest_1samp(sample, 10)) # гипотеза не верна
taxi_mex['pick_datetime'] = pd.to_datetime(taxi_mex.pickup_datetime)
taxi_mex['month'] = taxi_mex['pick_datetime'].dt.month
#print(stats.stats.ttest_ind(taxi_mex['trip_duration'].sample(n=3000),
# taxi_bog['trip_duration'].sample(n=3000))) # гипотеза об одинаковых поездках отвергается

#print(stats.stats.ttest_ind(taxi_mex['wait_sec'].sample(n=3000),
#      taxi_bog['wait_sec'].sample(n=3000))) #гипотеза об одинаковом ожидании подтвердиласьcь
control = taxi_mex[taxi_mex.month == 11]['trip_duration'].sample(n=1000)
treatment = taxi_mex[taxi_mex.month == 12]['trip_duration'].sample(n=1000)
print(stats.stats.ttest_rel(control, treatment)) #гипотеза подтверждена