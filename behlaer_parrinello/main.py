import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential


# DATA PREPERATION




ls = []
n = 64
for i in range(n):

    model = Sequential()

    model.add(Input(2))
    model.add(Dense(40))
    model.add(Dense(40))
    model.add(Dense(1))


    model.compile(optimizer='adam', loss='rmse')
    model.fit()

    ls.append(model())

total_energy = sum(ls)