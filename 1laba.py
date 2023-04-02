import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from random import randint


BIT_COUNT = 16
TOTAL_COUNT = 2**BIT_COUNT
LEARNING_SAMPLE_SIZE = 10000


def chertochka(ind):
    chislo = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in ind:
        chislo[i] = 1
    return chislo


def pocht_cod(x):
    mass = []
    if x == 0:
        ind = [0, 1, 2, 3, 4, 5, 6, 7, 8]
        mass = chertochka(ind)
    elif x == 1:
        ind = [1, 2, 6]
        mass = chertochka(ind)
    elif x == 2:
        ind = [0, 1, 3, 8]
        mass = chertochka(ind)
    elif x == 3:
        ind = [0, 6, 7, 8]
        mass = chertochka(ind)
    elif x == 4:
        ind = [1, 2, 5, 7]
        mass = chertochka(ind)
    elif x == 5:
        ind = [0, 2, 3, 5, 7]
        mass = chertochka(ind)
    elif x == 6:
        ind = [2, 3, 4, 6, 7]
        mass = chertochka(ind)
    elif x == 7:
        ind = [0, 4, 6]
        mass = chertochka(ind)
    elif x == 8:
        ind = [0, 1, 2, 3, 4, 5, 7]
        mass = chertochka(ind)
    elif x == 9:
        ind = [0, 1, 5, 7, 8]
        mass = chertochka(ind)

    return mass


def fit_len(d, count=BIT_COUNT):
    if len(d) < count:
        d = [0]*(count - len(d)) + d
    return d


x = []
y = []

for i in range(100):
    h = randint(0, 10)
    x.append(pocht_cod(h))
    y.append(h % 2)



x = np.array(x)
y = np.array(y)





x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1.0 - LEARNING_SAMPLE_SIZE/TOTAL_COUNT, random_state=42)

classes = ['четное', 'нечетное']
nb_classes = len(classes)

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
Y = np_utils.to_categorical(y, nb_classes)

model = Sequential()
model.add(Dense(input_dim=BIT_COUNT, activation="softmax", units=nb_classes))

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, Y_train, batch_size=128, epochs=5, verbose=2,
          validation_data=(x_test, Y_test))

model.save("my_model.h5")

score = model.evaluate(x_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])