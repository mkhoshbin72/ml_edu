import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model, Sequential

import numpy as np

from utils import SymetryFunctions, read_data


dataset = read_data('sample1.json')
sf = SymetryFunctions(
    dataset=dataset,
    R_c=1,
    R_s=1,
    eta=1,
    la=1,
    zeta=0.1,
)
g1 = sf.G1_sum()
g2 = sf.G2_sum()
data = np.array([(g1[0][i], g2[0][i]) for i in range(len(g1[0]))]).astype(np.float32)

targets = np.array(sf.get_targets()).astype(np.float32)

atoms_num = 64
symetry_funcs_num = 2

model = Sequential()
model.add(Input(shape=symetry_funcs_num))
model.add(Dense(40, activation='tanh'))
model.add(Dense(40, activation='tanh'))
model.add(Dense(1, activation='tanh'))
model.compile(optimizer='adam', loss=tf.keras.losses.MSE)

weights = model.get_weights()

ls = []
for i in range(atoms_num):

    model.fit(
        x=np.array([data[i, :]]),
        y=np.array(targets),
        batch_size=1,
        epochs=10
    )

    ls.append(model.predict(np.array([data[i, :]])))
    model.set_weights(weights)
    
print(ls)
# total_energy = sum(ls)