import cv2
from matplotlib import pyplot as plt
import numpy as np
from img_preprocess import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.utils import np_utils
from keras import backend as K
from sklearn.model_selection import train_test_split
np.random.seed(1337)  # for reproducibility
import sys

if __name__ == '__main__':
    img_rows, img_cols = 50, 50 # input image dimensions
    folder_list = ['imgs/rockler_bubinga', 'imgs/rockler_cherry', 'imgs/cs_sycamore']
    X, y = prep_total_pipeline(folder_list, img_rows)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    batch_size = 100
    num_classes = len(folder_list)
    num_epochs = 4

    nb_filters = 10 # number of convolutional filters to use
    pool_size = (5, 5) # size of pooling area for max pooling
    # decreases image size, and helps to avoid overfitting
    kernel_size = (10, 10) # slides over image to learn features

    # reshape image for Keras, note that image_dim_ordering set in ~.keras/keras.json
    # don't change any of this
    if K.image_dim_ordering() == 'th':
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[3], img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[3], img_rows, img_cols)
        input_shape = (X_test.shape[3], img_rows, img_cols)
    else:
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, X_train.shape[3])
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, X_test.shape[3])
        input_shape = (img_rows, img_cols, X_test.shape[3])

    # don't change conversion or normalization
    X_train = X_train.astype('float32') # data was uint8 [0-255]
    X_test = X_test.astype('float32')  # data was uint8 [0-255]
    X_train /= 255 # normalizing (scaling from 0 to 1)
    X_test /= 255  # normalizing (scaling from 0 to 1)
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices (don't change)
    Y_train = np_utils.to_categorical(y_train, num_classes) # cool
    Y_test = np_utils.to_categorical(y_test, num_classes)   # cool * 2
    # in Ipython you should compare Y_test to y_test

    model = Sequential() # model is a linear stack of layers (don't change)

    # note: the convolutional layers and dense layers require an activation function
    # see https://keras.io/activations/
    # and https://en.wikipedia.org/wiki/Activation_function
    # options: 'linear', 'sigmoid', 'tanh', 'relu', 'softplus', 'softsign'

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                        padding='valid',
                        input_shape=input_shape)) #first conv. layer (keep layer)
    model.add(Activation('tanh')) # Activation specification necessary for Conv2D and Dense layers

    model.add(MaxPooling2D(pool_size=pool_size)) # decreases size, helps prevent overfitting
    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Flatten()) # necessary to flatten before going into conventional dense layer (keep layer)
    print('Model flattened out to ', model.output_shape)

    # now start a typical neural network
    model.add(Dense(640)) # (only) 32 neurons in this layer, really?  (keep layer)
    model.add(Activation('relu'))
    model.add(Dropout(0.5)) # zeros out some fraction of inputs, helps prevent overfitting

    model.add(Dense(320))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_classes)) # 10 final nodes (one for each class) (keep layer)
    model.add(Activation('softmax')) # keep softmax at end to pick between classes 0-9

    # many optimizers available
    # see https://keras.io/optimizers/#usage-of-optimizers
    # suggest you keep loss at 'categorical_crossentropy' for this multiclass problem,
    # and metrics at 'accuracy'
    # suggest limiting optimizers to one of these: 'adam', 'adadelta', 'sgd'
    # how are we going to solve and evaluate it:
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # during fit process watch train and test error simultaneously
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs,
              verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1]) # this is the one we care about
