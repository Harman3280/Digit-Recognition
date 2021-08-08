
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import optimizers
from keras.datasets import mnist
from keras.utils import to_categorical
import keras


# ### loading mist hand written dataset

(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)




# ## Applying threshold for removing noise

_,X_train_th = cv2.threshold(X_train,127,255,cv2.THRESH_BINARY)
_,X_test_th = cv2.threshold(X_test,127,255,cv2.THRESH_BINARY)



# ### Reshaping

X_train = X_train_th.reshape(-1,28,28,1)
X_test = X_test_th.reshape(-1,28,28,1)


# ### Creating categorical output from 0 to 9 (One Hot Encoding)

y_train = to_categorical(y_train, num_classes = 10)
y_test = to_categorical(y_test, num_classes = 10)


# ## cross checking shape of input and output

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# # Creating CNN model

input_shape = (28,28,1)
number_of_classes = 10

model = Sequential()
# Adding Convolution layer with 32 Filters to extract features
model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))

# Adding Convolution layer with 64 Filters to extract features from prv extracted features
model.add(Conv2D(64, (3, 3), activation='relu'))

# Adding Pooling layer to reduce the size of Feature map ultimately reduces computation
model.add(MaxPool2D(pool_size=(2, 2)))

# adding dropout st our model will not learn redundant details , so it will work for variety of inputs
# Flattening the dataset to feed it too Dense layers
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
# Using "softmax" normalizing our probabilities (Comb Prob of All Outputs is 1)
model.add(Dense(number_of_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train,epochs=5, shuffle=True,
                    batch_size = 200,validation_data= (X_test, y_test))

# An H5 file is a data file saved in the Hierarchical Data Format (HDF).
# It contains multidimensional arrays of scientific data.
model.save('digit_classifier2.h5')
