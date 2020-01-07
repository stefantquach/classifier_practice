from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D, Add, ZeroPadding2D
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.initializers import glorot_uniform

import matplotlib.pyplot as plt
import numpy as np
import os

classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

batch_size = 128
epochs = 10
num_classes = 10

num_predictions = 20

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

def plot_image(prediction_array, true_label, image):
    prediction = classes[np.argmax(prediction_array)]
    true_value = classes[np.argmax(true_label)]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(image)
    if prediction == true_value:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(str(prediction),
                                100*np.max(prediction_array),
                                str(true_value)),
                                color=color)

def plot_value_array(prediction_array, true_label):
    prediction = classes[np.argmax(prediction_array)]
    true_value = classes[np.argmax(true_label)]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])

    thisplot = plt.bar(range(10), prediction_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)

    thisplot[predicted_label].set_color('red')
    thisplot[np.argmax(true_label)].set_color('blue')


def identity_block(X, f, filters, stage, block):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    # First component of main path
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    # X = Activation('relu')(X)
    X = Dropout(0.3)(X)

    # Second component of main path
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1, 1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)
    # X = Activation('relu')(X)

    # # Third component of main path
    # X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1, 1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):

    # Defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    # X = Activation('relu')(X)

    X = Dropout(0.3)(X)
    # Second component of main path
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)
    # X = Activation('relu')(X)

    # # Third component of main path
    # X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding='valid', name=conv_name_base + '2c', kernel_initializer=glorot_uniform(seed=0))(X)
    # X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    # X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)
    # X_shortcut = Activation('relu')(X)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1, 1), strides=(s, s), padding='valid', name=conv_name_base + '1', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X

def ResNet50(input_shape = (32, 32, 3), classes = 10, k=1):

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)


    # Zero-Padding
    # X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(16, (3, 3), strides = (1, 1), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X_input)
    # X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    filters = np.array([32, 32, 32])*k
    # Stage 2
    X = convolutional_block(X, f = 3, filters = filters, stage = 2, block='a', s = 1)
    X = identity_block(X, 3, filters, stage=2, block='b')
    X = identity_block(X, 3, filters, stage=2, block='c')
    X = identity_block(X, 3, filters, stage=2, block='d')

    filters *= 2
    # Stage 3
    X = convolutional_block(X, f=3, filters= filters, stage=3, block='a', s=2)
    X = identity_block(X, 3, filters, stage=3, block='b')
    X = identity_block(X, 3, filters, stage=3, block='c')
    X = identity_block(X, 3, filters, stage=3, block='d')

    filters *= 2
    # Stage 4
    X = convolutional_block(X, f=3, filters=filters, stage=4, block='a', s=2)
    X = identity_block(X, 3, filters, stage=4, block='b')
    X = identity_block(X, 3, filters, stage=4, block='c')
    X = identity_block(X, 3, filters, stage=4, block='d')
    #
    # # Stage 5
    # X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block='a', s=2)
    # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='b')
    # X = identity_block(X, 3, [512, 512, 2048], stage=5, block='c')

    # AVGPOOL
    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = AveragePooling2D(pool_size=(2,2), padding='same')(X)

    # Output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)

    # Create model
    model = keras.models.Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


model = ResNet50(input_shape = (32, 32, 3), classes = 10, k=2)
model.summary()
opt_rms = keras.optimizers.RMSprop(learning_rate=0.001, decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit(x_train[0:1], y_train[0:1], epochs=1)

#data augmentation
datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
    )
datagen.fit(x_train)

def learning_rate_schedule(epoch):
    if epoch < 50:
        return 0.01
    else:
        return 0.01/(5*int(epoch/50))

callback = keras.callbacks.LearningRateScheduler(learning_rate_schedule)

batch_size = 32
epochs=15
opt_rms = keras.optimizers.RMSprop(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test,y_test),
                    callbacks=[callback])
model.save_weights('cifar10_resnet_rms_ep75.h5')
