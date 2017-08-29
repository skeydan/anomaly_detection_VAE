from keras.models import Sequential
from keras.layers import Dense, Activation
from keras import backend as k
import numpy as np
import tensorflow as tf


model = Sequential()
model.add(Dense(4, input_dim=2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')
model.summary()

outputTensor = model.output 
listOfVariableTensors = model.trainable_weights
gradients = k.gradients(outputTensor, listOfVariableTensors)


trainingExample = np.random.random((1,2))
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
evaluated_gradients = sess.run(gradients,feed_dict={model.input:trainingExample})
evaluated_gradients
