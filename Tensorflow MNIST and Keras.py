# Tensorflow MNIST


import tensorflow as tf2
import matplotlib.pyplot as plt
import input_data as i_d


tf = tf2.compat.v1
tf.disable_v2_behavior()

"""
mnist = i_d.read_data_sets('/tmp/data/',one_hot=True)

image = mnist.train.images[9].reshape([28, 28])
#plt.gray()
#plt.imshow(image)
#plt.show()
#print(mnist.train.images[7].shape)
#print(mnist.train.labels[7].shape)



learning_rate = 0.1

epochs = 1000
# object per one learning time
# batch - part of dataset if dataset is too big
batch_size = 128 # the number of training objects in one batch

n_hidden_1 = 256 # number of neurons on 1 hidden layer
n_hidden_2 = 256 # number of neurons on 2 hidden layer
num_input = 784 # 28 x 28 (input vector)
num_classes = 10 # output matrix


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

        batch_x, batch_y = mnist.train.next_batch(batch_size)
        sess.run(train, feed_dict={X: batch_x, Y: batch_y})

        if epoch % 50 == 0:
            train_accuracy = sess.run(
                accuracy,
                feed_dict={
                    X: mnist.train.images,
                    Y: mnist.train.labels
                } )
            print('Epoch #{}: train accuracy = {}'.format(epoch, train_accuracy))

    print('Test accuracy = {}'.format(
        sess.run(
            accuracy,
            feed_dict={
                X:mnist.test.images,
                Y:mnist.test.labels,
            }
        )
    ))
# train accuracy = 0.821
"""

# Keras MNIST # easily then previous way

batch_size = 128
num_classes = 10
epochs = 1000


(x_train, y_train), (x_test, y_test) = tf2.keras.datasets.mnist.load_data()

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


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
y_train = tf2.keras.utils.to_categorical(y_train, num_classes)
y_test = tf2.keras.utils.to_categorical(y_test, num_classes)

print(y_train.shape[0], 'answer train')
print(y_test.shape[0], 'answer test')

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
