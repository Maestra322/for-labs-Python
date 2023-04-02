import numpy as np
import random
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split


A = -5 # начало интервала по x для выборки
B = 2 # конец интервала по x для выборки
TOTAL_COUNT = 1000 # всего интервалов
LEARNING_SAMPLE_SIZE = 200 # количество точек в выборке
EPOCH_COUNT = 200 # максимальное количество эпох
NEIRON_COUNT = 20 # количество нейронов в скрытом слое

x = np.array([A + (B - A)*i/TOTAL_COUNT for i in range(TOTAL_COUNT + 1)])

def f(x):
    answer = x**2 + 3*x - 4
    return answer

y = np.array([f(i) for i in x])
чп
plt.plot(x, y, label='функция')
ax = plt.gca()
ax.axhline(y=0, color='k') # рисуем ось X
ax.axvline(x=0, color='k') # рисуем ось Y
plt.legend() # добавляем легенду
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1.0 -
LEARNING_SAMPLE_SIZE/TOTAL_COUNT, random_state=42)

x_train.shape
y_train.shape

z = zip(x_train, y_train)
zs = sorted(z, key=lambda tup: tup[0])

x1 = [z[0] for z in zs]
y1 = [z[1] for z in zs]

plt.plot(x, y, label='Исходные')
plt.plot(x1, y1, label='Обучающие')
ax = plt.gca()
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.legend()
plt.show()

x_mean = x_train.mean(axis=0) # Среднее значение
x_std = x_train.std(axis=0) # Стандартное отклонение

print('Среднее значение:', x_mean)
print('Стандартное отклонение:', x_std)

x_train -= x_mean
x_train /= x_std
x_test -= x_mean
x_test /= x_std

print('Среднее значение после нормализации:', x_train.mean(axis=0))
print('Стандартное отклонение после нормализации:', x_train.std(axis=0))

model = Sequential()
model.add(Dense(NEIRON_COUNT, activation='relu', input_shape=(1,)))
model.add(Dense(NEIRON_COUNT, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary() # печатаем свойства модели

history = model.fit(x_train, y_train, batch_size=20, epochs=EPOCH_COUNT,
verbose=2, validation_data=(x_test, y_test))

plt.plot(history.history['mae'], label='Обучающая выборка')
plt.plot(history.history['val_mae'], label='Тестовая выборка')
plt.xlabel('Эпоха обучения')
plt.ylabel('Средняя абсолютная ошибка')
plt.legend()
plt.show()

mse, mae = model.evaluate(x_test, y_test, verbose=0)
print("Среднеквадратичное отклонение:", mse)
print("Средняя абсолютная ошибка:", mae)

x_norm = x - x_mean
x_norm /= x_std

y_pred = model.predict(x_norm)
print('Среднее значение после нормализации:', x_norm.mean(axis=0))
print('Стандартное отклонение после нормализации:', x_norm.std(axis=0))

plt.plot(x, y, label='Исходные')
plt.plot(x, y_pred, label='Прогнозные')
ax = plt.gca()
ax.axhline(y=0, color='k')
ax.axvline(x=0, color='k')
plt.legend()
plt.show()

TRAIN_M = 1.6
TRAIN_SKO = 0.9


def variate(z, m, sko):
    return z * random.normalvariate(m, sko)


y_train = np.array([variate(i, TRAIN_M, TRAIN_SKO) for i in y_train])

TEST_M = 1.6
TEST_SKO = 0.9
y_test = np.array([variate(i, TEST_M, TEST_SKO) for i in y_test])

early_stopping_callback = EarlyStopping(monitor='val_mae', patience=3)

history = model.fit(x_train, y_train, batch_size=20, epochs=EPOCH_COUNT,
                    verbose=2, validation_data=(x_test, y_test),
                    callbacks=[early_stopping_callback])
print("Обучение остановлено на эпохе", early_stopping_callback.stopped_epoch)
