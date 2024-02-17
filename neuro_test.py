from tensorflow import keras
from keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Sequential

import pandas, numpy
model = Sequential()
model.add(Input(5))
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=32, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(1, activation="relu"))
print(model.summary())

model.compile(loss='mse', metrics=['mse'])
model = keras.models.load_model('models/model_main')

