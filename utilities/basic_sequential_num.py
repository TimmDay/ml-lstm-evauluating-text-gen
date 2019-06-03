# from tutorial
# https://www.youtube.com/watch?v=iMIWee_PXl8

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# Prepare Data
# /100 for normalization
data = [[[(i+j)/100] for i in range(5)] for j in range(100)] # sequences of numbers of length 5 (eg. 0,1,2,3,4 )
target = [(i+5)/100 for i in range(100)] # the next digit (target) for each sequence (eg. 5)

# print(data)
# print(target)

#  convert to numpy data format for input to LSTM
data = np.array(data, dtype=float)
target = np.array(target, dtype=float)

# print(data)
# print(data.shape)
# print(target.shape)

x_train, x_test, y_train, y_test = train_test_split(data,target, test_size=0.2, random_state=4)
# print(x_test) # x is input data
# print(y_test) # y is target




#  LSTM Model
model=Sequential()
# add layer -> output layer
model.add(LSTM((1),batch_input_shape=(None, 5, 1),return_sequences=True)) # specify output layer size, describe input shape (use None if unknown)
model.add(LSTM((1),return_sequences=True))
# model.add(LSTM((1),return_sequences=True))
# model.add(LSTM((1),return_sequences=True))
model.add(LSTM((1),return_sequences=False))
# number of inputs, input length 5, each vector in input length 1
# return_sequences -> True: return output after each node, False: return output only after final node. False for us since target has a single digit
# if multilayer, you need to output at each step iot pass to next step
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
model.summary()

# fit the training data
# return history to get a visualisation - useful in visualising loss

history = model.fit(x_train, y_train, epochs=300, validation_data=(x_test,y_test))

# why the poor results? plot to investigate
results = model.predict(x_test)

plt.scatter(range(20),results,c='r') # red color for model predicted results
plt.scatter(range(20),y_test,c='g') # green for gold standard
plt.show()

plt.plot(history.history['loss'])
plt.show()

model.summary()

# we should stop the model when the loss has converged. ie when it hastopped decreasing. if you stop it early, increase the epochs to run it longer

# with a second layer, the results are much better and the loss function levels out much earlier

# what are the parameters? two layers has 24, but needs more epochs to converge.
# 3 layers has 36, and converges much faster
# however, increasing to 5 layers killed the accuracy. 4 also sucked. 3 lost accuracy for high numbers but was good for low

# RNNS and LSTMs work best when we can throw variable sized inputs at them
# - pad with zeroes so all inputs the same length
# - make sequence size none and batch size 1
#   - - changelike this batch_input_shape=(None, None, 1), then have variable sized inputs (try changing x input length to 6 and y to 7, still decent)
#   - - data = [[[(i+j)/100] for i in range(6)] for j in range(100)] # sequences of numbers of length 5 (eg. 0,1,2,3,4 )
#   - - target = [(i+6)/100 for i in range(100)]
# - group data by same sized inputs and train in groups
