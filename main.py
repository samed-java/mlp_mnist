import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.datasets import mnist
from keras.optimizers import rmsprop_v2
import matplotlib.pyplot as plt

(x_tr, y_tr), (x_test, y_test) = mnist.load_data()
a = x_test[1]
num_classes = 10
batch_size = 128
epochs = 150

x_tr = x_tr.reshape(60000, 784)
x_test = x_test.reshape(10000,784)
x_tr = x_tr.astype('float32')
x_test = x_test.astype('float32')
x_tr=x_tr/255
x_test=x_test/255

y_tr = keras.utils.np_utils.to_categorical(y_tr, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(784, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(392, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer=rmsprop_v2.RMSProp(), metrics=['accuracy'])

history = model.fit(x_tr, y_tr, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
prediction = model.predict(x_test)
print(y_test[1])
plt.imshow(a)
plt.plot(prediction, y_test, 'o')
plt.show()
