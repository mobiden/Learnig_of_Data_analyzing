import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12,5)

df = pd.read_csv('food.txt', sep=' ')

#перенос всех объектов с признаками в матрицу Х
X = df.iloc[:, 1:].values
#вычитает из матрицы среднее значение по всем признакам
# и делит на стандартное отклонение по всем признакам
X = (X - X.mean(axis = 0))/X.std(axis=0)

#расчет среднего значение результирующей матрицы: близко к нулю
X.mean(axis=0)
#расчет стандартного отклонения - равны 1
X.std(axis=0)

from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# method -пересчет расстояние между кластерами, metric - расчет попарного расстояния
Z = linkage(X, method='average', metric='euclidean')
d = dendrogram(Z, orientation='left', labels=df.Name)
label = fcluster(Z, 2.2, criterion='distance') # 2.2 - порог отсечения
np.unique(label) # количество кластеров по верхнему критерию
df.loc[:, 'label'] = label #создание в датафрейме нового столбца с label

for i, group in df.groupby('label'): #таблица разбиения по кластерам
    print('=' * 10)
    print('cluster{}'.format(i))
    print(group)
