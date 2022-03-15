import numpy as np
from numpy.random._examples.numba.extending_distributions import x

х = np.array([1, 2, 3, 4, ], dtype='int64')
y = x[0]
x.shape # размерность массива
k = np.ones(5)
k = np.zeros(5)
k = np.eye(5) # диагональная матрица с единицами
k = np.random.random((2, 3))

m = np.array([[2, 3, 4, 6, 7],
              [5, 6, 7, 99, 1]]
             )
s = m[0, 2]
t = m[:,:3] # все значения по строкам и в первых трех столбцах
u = m > 2 #маска с bool по условию
f = m[m > 2] #значения по условию
a = m.flatten() # превратить в одномерный массив
m.T #транспонировать
m.reshape((10,1)) # не изменяет массив
m.resize((10,1)) # изменяет массив
skalyar = np.dot(m, t)
