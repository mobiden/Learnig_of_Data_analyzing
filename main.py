# Tensorflow MNIST

import pandas as pd
import tensorflow as tf2
import matplotlib.pyplot as plt
import numpy as np


def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels.
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


tf = tf2.compat.v1
tf.disable_v2_behavior()


tr_mnist = pd.read_csv('Fashion_MNIST/fashion-mnist_train.csv')
te_mnist = pd.read_csv('Fashion_MNIST/fashion-mnist_test.csv')
y_train = tr_mnist.iloc[:, :1].values
x_train = tr_mnist.iloc[:, 1:].values
y_test = te_mnist.iloc[:, :1].values
x_test = te_mnist.iloc[:, 1:].values

temp_y = []
for y in y_train:
    my_list = [0 * i for i in range(10)]
    my_list[y[0]] = 1
    temp_y.append(my_list)
y_train = temp_y

temp_y = []
for y in y_test:
    my_list = [0 * i for i in range(10)]
    my_list[y[0]] = 1
    temp_y.append(my_list)
y_test = temp_y



image = x_train[5].reshape([28, 28])
plt.gray()
plt.imshow(image)
plt.show()


learning_rate = 0.1

epochs = 1000
# object per one learning time
# batch - part of dataset if dataset is too big
batch_size = 128 # the number of training objects in one batch

n_hidden_1 = 256 # number of neurons on 1 hidden layer
n_hidden_2 = 256 # number of neurons on 2 hidden layer
num_input = 784 # 28 x 28 (input vector)
num_classes = 10 # output matrix


#X = tf.placeholder('float', [None, num_input]) #num_input - data dimension
#Y = tf.placeholder('float', [None, num_classes])
X = tf.placeholder('float', [None, num_input]) #num_input - data dimension
Y = tf.placeholder('float', [None, num_classes])

# neural network configuration

weights = {
    # random initialization of weights in neurons
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'output': tf.Variable(tf.random_normal([n_hidden_2, num_classes])),
           }

biases = {
# random initialization of biases in neurons
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'output': tf.Variable(tf.random_normal([num_classes])),
}
def network(x): # function of relation between neurons
    # multiplying weights and adding constants on each layer
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    output_layer = tf.matmul(layer2, weights['output']) + biases['output']

    return output_layer

logits = network(X)

# loss function for distribution by superclasses (logits)
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits,
        labels=Y,
                                              ))
# Adam optimization engine
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

# search for closest prediction
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()


# neural net learning

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):

        batch_x, batch_y = next_batch(batch_size, x_train, y_train)


        sess.run(train, feed_dict={X: batch_x, Y: batch_y})

        if epoch % 50 == 0:
            train_accuracy = sess.run(
                accuracy,
                feed_dict={
                    X: x_train,
                    Y: y_train,
                } )
            print('Epoch #{}: train accuracy = {}'.format(epoch, train_accuracy))

    print('Test accuracy = {}'.format(
        sess.run(
            accuracy,
            feed_dict={
                X: x_test,
                Y:y_test,
            }
        )
    ))
# train accuracy = 0.821



# Keras MNIST # easily then previous way
print ('Keras MNIST')
batch_size = 128
num_classes = 10
epochs = 5





x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# нормализация значение пикселей
x_train /= 255
x_test /= 255

print(x_train.shape[0], 'количество train samples')
print(x_test.shape[0], 'test samples')

# перевод в one-hot представление
#y_train = tf2.keras.utils.to_categorical(y_train, num_classes)
#y_test = tf2.keras.utils.to_categorical(y_test, num_classes)
y_train = np.array(y_train)
y_test = np.array(y_test)

#print(y_train.shape[0], 'answer train')
#print(y_test.shape[0], 'answer test')

model = tf2.keras.models.Sequential() # определяем модель - будет последовательностью слоев
# первый слой из 512 нейронов
model.add(tf2.keras.layers.Dense(512, activation='relu', input_shape=(784,)))
# Dropout выкидывает случайные нейроны во время тренировки для улучшения обучения
model.add(tf2.keras.layers.Dropout(0.2))
model.add(tf2.keras.layers.Dense(512, activation='relu'))
model.add(tf2.keras.layers.Dropout(0.2))

# выходной слой размерности num_classes. softmax для логистической регрессии
model.add(tf2.keras.layers.Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=tf2.keras.optimizers.Adam(),
    metrics=['accuracy'],
)
_ = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose= 1,
    validation_data=(x_test, y_test)
)
