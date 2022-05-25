# Tensorflow intro

import tensorflow as tf2
tf = tf2.compat.v1

tf.disable_v2_behavior()

"""
hello = tf.constant('Hello world!')
#result = hello # для TF2
#tf.print(result)

with tf.Session() as sess: # Сессии для версии 1 Тензора
    result = sess.run(hello)
    print(result)


a = tf.constant(2)
b = tf.constant(3)
c = tf.constant([1,2,3,4])
d = tf.constant([2,3,4,5])

with tf.Session() as sess:
    print('a = {},b = {}, c={}, d={}'.format(
    sess.run(a), sess.run(b), sess.run(c), sess.run(d)))
    print('a+b={}\n'
        'a*b={}'.format(sess.run(a+b), sess.run(a*b)))
    print('c+d={}\n'
        'c*d={}'.format(sess.run(c+d), sess.run(c*d)))


# placeholder - добавление переменной в граф
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

# добавление операций
add = tf.add(a,b)
mul = tf.multiply(a,b)

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs', sess.graph)
    # > tensorboard --logdir logs/
    print("a+b={}".format(sess.run(add, feed_dict={a: 3, b:1})))
    print("a*b={}".format(sess.run(mul, feed_dict={a: 7, b:8})))
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression # случайные данные

n_samples = 42
x_train, y_train = make_regression(n_samples=n_samples, n_features=1,
                                   noise=15, random_state=7)
# нормализация модели
x_train = (x_train - x_train.mean()) / x_train.std()
y_train = (y_train - y_train.mean()) / y_train.std()
#print(x_train[:5]) # точки на прямой

X = tf.placeholder('float')
Y = tf.placeholder('float')

# создание случайных переменных
W = tf.Variable(np.random.randn(), name="weight")
b = tf.Variable(np.random.randn(), name="bias")

prediction = tf.add(tf.multiply(W, X), b) # получаем вершину в графе (ноду), которая
                                          #  соответствует получению предсказаний

# переменная, отвечающие за изменение весов
learning_rate = tf.placeholder(tf.float32, shape=[])

# определяем функцию потерь - минимизируем квадратичную функцию ошибки
cost = tf.reduce_sum(tf.pow(prediction - Y, 2)) / n_samples

# обучение с помощью градиентного спуска и минимизацией функции потерь
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# инициализация глобальных переменных - нужно всегда делать.
init  = tf.global_variables_initializer()

epochs = 1000
sess = tf.Session()
sess.run(init) # инициализация переменных

lr = 0.1 # первоначально зададим learning_rate = 0.1
for epoch in range(epochs):
    for (x_batch, y_batch) in zip(x_train, y_train):
        sess.run(optimizer, feed_dict={X: x_batch, Y: y_batch, learning_rate:lr})
    if epoch % 100 == 0: # выводим отладочную информацию каждую сотую эпоху
        lr /= 2
        c = sess.run(cost, feed_dict={X: x_train, Y:y_train})
 #       print("Epoch #{}: cost:{}".format(epoch, c))

plt.plot(x_train, y_train, 'ro', label="original data")
plt.plot(x_train, sess.run(W) * x_train + sess.run(b), label="fitted line")
plt.legend()
plt.show()
sess.close()
