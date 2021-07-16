# Info:
# Convolutional Neural Network

# Author: Elaine Chen, Jason Liu
# Date: 7/11/2021

# Purpose:Project & Education

# inputs: Cifar-10 & Cifar-100

# outputs: Benchmarking

# Version control:

#-----------------------------------------------------------------------------------------------------------------------------------------------------------
# import libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot 
from keras.datasets import cifar100
from keras.models import Sequential 
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import BatchNormalization

# Import the cifar100 dataset
(x_train_all, y_train_all), (x_test_all, y_test_all) = cifar100.load_data()

# Get some random pictures in the cifar-100 dataset
x_train_all_ran = x_train_all[1000:1300, :, :]
# Look at 6 of the example images:
for i in range(6):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(x_train_all_ran[i])
pyplot.show()

# data preprocessing:
# this is used for data conversion and normalization
train_norm = x_train_all.astype('float32')
test_norm = x_test_all.astype('float32')
# normalize to range 0-1
x_train_all = train_norm / 255.0
x_test_all = test_norm / 255.0
# return normalized images

# one hot encode labels
from keras.utils import to_categorical
y_train_all = to_categorical(y_train_all)
y_test_all = to_categorical(y_test_all)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------------------------------------------
# Trial 0: Find the best model
# Trial 0 -1 : (1 CNN layer)
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(128, (3, 3), activation='relu', input_shape=(image_size, image_size, 3)))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

# plot the architecture of the model
from keras.utils import plot_model
plot_model(classifier, to_file='trail0_1.png', show_shapes=True)

# Create a model checkpoint
from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)
# Start Training
history = classifier.fit(x_train_all, y_train_all, callbacks=[checkpointer], epochs=30, 
                         batch_size=30, 
                         validation_data=(x_test_all, y_test_all))

# model Accuracy and Loss
pyplot.title('Model Accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.plot(history.history['accuracy'])
pyplot.plot(history.history['val_accuracy'])
pyplot.legend(['train','test'], loc='upper left')
pyplot.show()
pyplot.title('Model Loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.legend(['train','test'], loc='upper left')
pyplot.show()

# -----------------------------------------------------------------------------
# Trial 0 -2 : (1 VGG Block)
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
classifier.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Flatten())
classifier.add(Dense(32, activation = 'relu'))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])


plot_model(classifier, to_file='trail0_1.png', show_shapes=True)

checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit(x_train_all, y_train_all, callbacks=[checkpointer], epochs=30, 
                         batch_size=30, 
                         validation_data=(x_test_all, y_test_all))
# model Accuracy and Loss
pyplot.title('Model Accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.legend(['train','test'], loc='upper left')
pyplot.show()

pyplot.title('Model Loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.legend(['train','test'], loc='upper left')
pyplot.show()

# Trial 0 -3 : (3 VGG Block)
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_size, image_size, 3)))
classifier.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu'))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

plot_model(classifier, to_file='trail0_1.png', show_shapes=True)

checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit(x_train_all, y_train_all, callbacks=[checkpointer], epochs=30, 
                         batch_size=30, 
                         validation_data=(x_test_all, y_test_all))
# model Accuracy and Loss
pyplot.title('Model Accuracy')
pyplot.ylabel('accuracy')
pyplot.xlabel('epoch')
pyplot.plot(history.history['acc'])
pyplot.plot(history.history['val_acc'])
pyplot.legend(['train','test'], loc='upper left')
pyplot.show()

pyplot.title('Model Loss')
pyplot.ylabel('loss')
pyplot.xlabel('epoch')
pyplot.plot(history.history['loss'])
pyplot.plot(history.history['val_loss'])
pyplot.legend(['train','test'], loc='upper left')
pyplot.show()


# -----------------------------------------------------------------------------
# Trial 1: Base + avg pooling in the transitional layers
# Convolution
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(image_size, image_size, 3)))
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(AveragePooling2D((2, 2), padding='same'))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

plot_model(classifier, to_file='trail0_1.png', show_shapes=True)

checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit(x_train_all, y_train_all, callbacks=[checkpointer], epochs=30, 
                         batch_size=30, 
                         validation_data=(x_test_all, y_test_all))

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 

# -----------------------------------------------------------------------------
# Trial 2: Base + 0.2 Dropouts 
# Convolution
from keras.layers import Dropout
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(image_size, image_size, 3)))
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(0.2))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

plot_model(classifier, to_file='trail0_1.png', show_shapes=True)

checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit(x_train_all, y_train_all, callbacks=[checkpointer], epochs=30, 
                         batch_size=30, 
                         validation_data=(x_test_all, y_test_all))

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 

# -----------------------------------------------------------------------------
# Trial 3: Base + Increasing Dropouts 
# Convolution
from keras.layers import Dropout
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(image_size, image_size, 3)))
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.3))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.4))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(0.4))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

plot_model(classifier, to_file='trail0_1.png', show_shapes=True)

checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit(x_train_all, y_train_all, callbacks=[checkpointer], epochs=30, 
                         batch_size=30, 
                         validation_data=(x_test_all, y_test_all))

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 

# -----------------------------------------------------------------------------
# Trial 4: Base + Weight Decay
# Convolution
from tensorflow.python.keras import regularizers
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005), input_shape=(image_size, image_size, 3)))
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

plot_model(classifier, to_file='trail0_1.png', show_shapes=True)

checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit(x_train_all, y_train_all, callbacks=[checkpointer], epochs=30, 
                         batch_size=30, 
                         validation_data=(x_test_all, y_test_all))

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 

# -----------------------------------------------------------------------------
# Trial 5: Base + data augmentation
# Convolution
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(image_size, image_size, 3)))
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    shear_range = 0.2,
    zoom_range = 0.2,
    featurewise_center=False,  
    samplewise_center=False, 
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    zca_whitening=False,  
    rotation_range=30,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True,  
    vertical_flip=False,
    fill_mode='nearest') 

datagen.fit(x_train_all)

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit_generator(datagen.flow(x_train_all, y_train_all, batch_size = 30),callbacks=[checkpointer], epochs=30, 
                                   validation_data=(x_test_all, y_test_all))

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 

# -----------------------------------------------------------------------------
# Trial 6: Base + Batch Normalization
# Convolution
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(image_size, image_size, 3)))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

plot_model(classifier, to_file='trail0_1.png', show_shapes=True)

checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit(x_train_all, y_train_all, callbacks=[checkpointer], epochs=30, 
                         batch_size=30, 
                         validation_data=(x_test_all, y_test_all))

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 

# -----------------------------------------------------------------------------
# Trial 7: Base + Increasing Dropouts + weight decay
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005), input_shape=(image_size, image_size, 3)))
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.3))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(0.0005)))
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.4))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(0.4))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

plot_model(classifier, to_file='trail0_1.png', show_shapes=True)

checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit(x_train_all, y_train_all, callbacks=[checkpointer], epochs=30, 
                         batch_size=30, 
                         validation_data=(x_test_all, y_test_all))

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 

# -----------------------------------------------------------------------------
# Trial 8: Base + 0.2 Dropouts + Batch Normalization + Data augmentation

classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(image_size, image_size, 3)))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.2))
classifier.add(Flatten())
classifier.add(Dense(128, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(0.2))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

datagen = ImageDataGenerator(
    shear_range = 0.2,
    zoom_range = 0.2,
    featurewise_center=False,  
    samplewise_center=False, 
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    zca_whitening=False,  
    rotation_range=30,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True,  
    vertical_flip=False,
    fill_mode='nearest') 

datagen.fit(x_train_all)

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit_generator(datagen.flow(x_train_all, y_train_all, batch_size = 30),callbacks=[checkpointer], epochs=30, 
                                   validation_data=(x_test_all, y_test_all))

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 

# -----------------------------------------------------------------------------
# Trial 9: Trial 9: VGG16 + Increasing Dropouts + Batch Normalization + Data augmentation + weight decay + 300 epochs
classifier = Sequential()
image_size=32
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(image_size, image_size, 3)))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Dropout(0.2))
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(AveragePooling2D((2, 2)))
classifier.add(Dropout(0.3))
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Dropout(0.3))
classifier.add(Convolution2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2), padding='same'))
classifier.add(Dropout(0.3))
classifier.add(Convolution2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Convolution2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D((2, 2)))
classifier.add(Dropout(0.3))
classifier.add(Flatten())
classifier.add(Dense(512, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(0.4))
classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

# image augmentation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    shear_range = 0.2,
    zoom_range = 0.2,
    featurewise_center=False,  
    samplewise_center=False, 
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    zca_whitening=False,  
    rotation_range=30,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True,  
    vertical_flip=False,
    fill_mode='nearest') 

datagen.fit(x_train_all)

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit_generator(datagen.flow(x_train_all, y_train_all, batch_size = 128),callbacks=[checkpointer], epochs=300, 
                                   validation_data=(x_test_all, y_test_all))

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 


# -----------------------------------------------------------------------------
# Trial 10: VGG16 + Increasing Dropouts + Batch Normalization + Data augmentation + weight decay + 300 epochs


classifier = Sequential()
image_size=32

classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(image_size, image_size, 3)))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.2))

classifier.add(Convolution2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())

classifier.add(MaxPooling2D((2, 2)))

classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.3))

classifier.add(Convolution2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())

classifier.add(AveragePooling2D((2, 2)))


classifier.add(Convolution2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))

classifier.add(Convolution2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))

classifier.add(Convolution2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())

classifier.add(MaxPooling2D((2, 2)))

classifier.add(Convolution2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))

classifier.add(Convolution2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))

classifier.add(Convolution2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())

classifier.add(MaxPooling2D((2, 2), padding='same'))

classifier.add(Convolution2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))

classifier.add(Convolution2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())
classifier.add(Dropout(0.4))

classifier.add(Convolution2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
classifier.add(BatchNormalization())

classifier.add(MaxPooling2D((2, 2)))
classifier.add(Dropout(0.5))

classifier.add(Flatten())
classifier.add(Dense(512, activation = 'relu', kernel_initializer='he_uniform'))
classifier.add(Dropout(0.5))

classifier.add(Dense(100, activation = 'softmax'))
classifier.summary()

# image augmentation
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
    shear_range = 0.2,
    zoom_range = 0.2,
    featurewise_center=False,  
    samplewise_center=False, 
    featurewise_std_normalization=False,  
    samplewise_std_normalization=False,  
    zca_whitening=False,  
    rotation_range=30,  
    width_shift_range=0.1,  
    height_shift_range=0.1,  
    horizontal_flip=True,  
    vertical_flip=False,
    fill_mode='nearest') 

datagen.fit(x_train_all)

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', 
                   metrics = ['accuracy'])

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath="cifar100_CNN.hdf5", 
                               monitor = 'val_acc',
                               verbose=1, 
                               save_best_only=True)

history = classifier.fit_generator(datagen.flow(x_train_all, y_train_all, batch_size = 128),callbacks=[checkpointer], epochs=300, 
                                   validation_data=(x_test_all, y_test_all))
#save the model
classifier.save("cifar100_VGG16.h5")
print("Saved model to disk")

pyplot.subplot(211)
pyplot.title('Loss')
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend(['train','test'], loc='upper right')
pyplot.subplot(212)
pyplot.title('Accuracy')
pyplot.plot(history.history['accuracy'], label='train')
pyplot.plot(history.history['val_accuracy'], label='test')
pyplot.legend(['train','test'], loc='upper left')
pyplot.subplots_adjust(bottom=-0.3) 

## Load the Model
from keras.models import load_model
model_name = "cifar100_VGG16.h5"
classifier = load_model(model_name)
## testing 
from keras.preprocessing import image
truck = image.load_img("pickup truck test.jpg", target_size=(32, 32))
truck = image.img_to_array(truck)
truck = truck.reshape(1, 32, 32, 3)
truck = truck.astype('float32')
truck = truck / 255.0

cloud = image.load_img("cloud test.jpg", target_size=(32, 32))
cloud = image.img_to_array(cloud)
cloud = cloud.reshape(1, 32, 32, 3)
cloud = cloud.astype('float32')
cloud = cloud / 255.0

leopard = image.load_img("leopardtest.jpg", target_size=(32, 32))
leopard = image.img_to_array(leopard)
leopard = leopard.reshape(1, 32, 32, 3)
leopard = leopard.astype('float32')
leopard = leopard / 255.0

# execute predictions
truck_pred = classifier.predict_classes(truck)
print(truck_pred[0])
cloud_pred = classifier.predict_classes(cloud)
print(cloud_pred[0])
leopard_pred = classifier.predict_classes(leopard)
print(leopard_pred[0])


