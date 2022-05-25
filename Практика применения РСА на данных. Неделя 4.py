import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 5)
import os
#os.environ["PROJ_LIB"] = "C:\\Python\\Anaconda\\pkgs\\proj-8.0.1-h1cfcee9_0\\Library\share\proj"

df_wine = pd.read_csv('winequality-red.csv', sep=',')
#print('Начальный датафрейм: ', df_wine.shape)


df_wine.loc[:, 'quality_cat'] = (df_wine.quality > 5).astype(int)
df_wine = df_wine.drop('quality', axis=1)
X = df_wine.iloc[:, : -1].values
Y = df_wine.iloc[:, -1].values

#метод главных компонентов

# при помощи внутренних функций sklearn
from sklearn.decomposition import PCA
pca = PCA(n_components=6)
pca.fit(X)
Z = pca.transform(X)
#print('Сжатый датафрейм Z: ',  Z)
#print('Коэффиценты сжатия: \n', pca.components_)
#print('Матрица коэффицентов: ', pca.components_.shape)

X_ = X - X.mean(axis=0) #центрируем признаки - вычитаем среднюю из всех
#print ('Сжатый вручную датафрейм: \n', X_.dot(pca.components_.T)) #сами умножаем матрицу Х на транспонированную (чтобы сошлись размерности) матрицу коэффицентов
# матрицы совпадают

# через сингулярное разложение
from numpy.linalg import svd

u, s, vt = svd(X_, full_matrices=0)
#print('размерности 3-х матриц: ', u.shape, s.shape, vt.shape)
S = np.diag(s) # для проверки вектор превращаем в матрицу
X_svd = u.dot(S).dot(vt) #собираем обратно датафрейм
#print('Вычислем погрешность между исходным и полученным DF: ',((X_ - X_svd) ** 2).sum())
#pca.components_ и vt совпадают по числам, но меняется знак иногда
v = vt[:6].T
Z_svd =X_.dot(v)
# Z и Z_svd совпадают по числам, но меняется иногда знак

# с помощью собственный чисел и векторов матрицы ковариации
from numpy.linalg import eig

C = X_.T.dot(X_) # расчет матрицы ковариации
lamb, W = eig(C) # собственные числа и матрица собственных векторов
# W совпадает с vt
#print('доля объясненной дисперсии по внутренней функции PCA: \n', pca.explained_variance_ratio_)   #
#print('доля объясненной дисперсии по матрице ковариации: \n', lamb/lamb.sum())

# Влияние количество компонент  на качество

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

model_baseline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression())
    ])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
cvs = cross_val_score(model_baseline, X, Y, scoring='accuracy', cv=cv).mean()
print ('Среднее значение меры качества: ', cvs)

scores = []
k = range(1, 12)
for n in k:
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n)),
        ('clf', LogisticRegression()),
        ])
    scores.append( cross_val_score(model, X, Y, scoring='accuracy', cv=cv).mean())
plt.plot(k, scores)
plt.hlines(cvs, 1, 12, colors='green')
# график показывает, что качество не сильно падает при уменьшении количества компонент
plt.show()