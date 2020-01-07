from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, AveragePooling2D, Add
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


def convolutional_block(X, f, filters, drop, s=2):

    F1, F2, F3 = filters

    X_shortcut = X

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(s,s), padding='valid',
                kernel_initializer = glorot_uniform(seed=0))(X)
    X = Dropout(drop)(X)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same',
               kernel_initializer = glorot_uniform(seed=0))(X)

    X_shortcut = BatchNormalization()(X_shortcut)
    X_shortcut = Activation('relu')(X_shortcut)
    X_shortcut = Conv2D(filters=F3, kernel_size=(1,1), strides=(s,s), padding='valid',
                        kernel_initializer = glorot_uniform(seed=0))(X_shortcut)

    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)

    return X


def make_group(X, N, outf, stride, drop):
    for i in range(N):
        X = convolutional_block(X, 3, [outf, outf, outf], s=(stride if i == 0 else 1), drop=drop)
    return X


def resNet(input_shape, num_classes, n_grps, N, k=1, drop=0.3, first_width=16):
    X_input = Input(input_shape)
    X = Conv2D(first_width, (3,3), padding='valid')(X_input)

    widths = [first_width]
    for grp in range(n_grps):
        widths.append(first_width*(2**grp)*k)

    for grp in range(n_grps):
        X = make_group(X, N, widths[grp+1], (1 if grp == 0 else 2), drop)

    X = BatchNormalization()(X)
    X = Activation('relu')(X)
    X = AveragePooling2D(pool_size=(1,1), padding='same')(X)
    X = Flatten()(X)
    X = Dense(num_classes, activation='softmax')(X)

    model = keras.models.Model(inputs = X_input, outputs=X)
    return model


model = resNet(input_shape=(32,32,3), num_classes=10, n_grps=3, N=4, k=1)
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

#training
batch_size = 64
epochs=25
opt_rms = keras.optimizers.RMSprop(learning_rate=0.01, decay=1e-6)
model.compile(loss='categorical_crossentropy',
        optimizer=opt_rms,
        metrics=['accuracy'])
model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),steps_per_epoch=x_train.shape[0] // batch_size, epochs=epochs,verbose=1,validation_data=(x_test,y_test))
model.save_weights('cifar10_normal_rms_ep75_1.h5')



predictions = model.predict(x_test)
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
print(classes)
random_index = np.random.randint(0, high=len(x_test)-1, size=num_images)
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(predictions[random_index[i]], y_test[random_index[i]], x_test[random_index[i]])
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(predictions[random_index[i]], y_test[random_index[i]])
plt.tight_layout()
plt.show()
