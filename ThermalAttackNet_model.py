import os
import glob
import sys

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

import PIL
from PIL import Image
import requests
from io import BytesIO
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, Flatten, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

#Verify that tensorflow is running with GPU -->
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#Get training lables -->
def list_dirs(path):
    return [os.path.basename(x) for x in filter(
        os.path.isdir, glob.glob(os.path.join(path, '*')))]

labels = list_dirs("Label/Directory/Training")
labels.sort()

epochs = 140

# Base directory of raw jpg/png images
base_dir = 'Your/Base/Directory'

train_path = os.path.join(base_dir, 'Training')
nb_train_samples = 1600
valid_path = os.path.join(base_dir, 'Test')
nb_val_samples = 400
test_path = os.path.join(base_dir, 'test-multiple_pics')

batch_size = 10

train_steps = nb_train_samples // batch_size
valid_steps = nb_val_samples // batch_size

classes = labels

img_width, img_height = 224, 224

train_datagen = ImageDataGenerator(rescale=1. / 255)
# This is the augmentation configuration we will use for training when data_augm_enabled argument is True
train_augmented_datagen = ImageDataGenerator(
                    rescale=1. / 255,
                    rotation_range=40,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

num_classes = len(classes)
augmented_train_generator = train_augmented_datagen.flow_from_directory(train_path, target_size=(img_width, img_height),
                                                                        classes=classes, class_mode='categorical',
                                                                        batch_size=batch_size)

val_datagen = ImageDataGenerator(rescale=1. / 255)

val_generator = val_datagen.flow_from_directory(valid_path, target_size=(img_width, img_height),
                                                    classes=classes, class_mode='categorical',
                                                    batch_size=batch_size)

# ModelCheckpoint
filepath = "/Your/Directory/model_thermalattacknet.h5"
checkpointer = ModelCheckpoint(filepath=filepath,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='auto',
                                   period=1,
                                   save_weights_only=True)
# Early stops based on loss
early_stop = EarlyStopping(monitor='val_loss', patience=3, mode='min')
#log_filepath = base_dir + '/model_training_log.csv'
log_filepath = 'thermalattacknet_log.csv'
csv_logger = CSVLogger(log_filepath, append=True, separator=',')

#callbacks_list = [early_stop, csv_logger]
callbacks_list = [checkpointer, early_stop, csv_logger]

model = keras.Sequential(name="thermal_attack")
model.add(keras.Input(shape=(224, 224, 3)))  # 224x224 RGB images
model.add(layers.Conv2D(32, 5, strides=2, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(3))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.Conv2D(32, 3, activation="relu"))
model.add(layers.MaxPooling2D(2))
model.add(layers.GlobalMaxPooling2D())
model.add(layers.Dense(4, activation='softmax', name='PREDICTIONS'))

model.summary()

model.summary()

history = model.fit_generator(train_generator, epochs= epochs,
                                      steps_per_epoch=train_steps,
                                      validation_data = val_generator,
                                      validation_steps= valid_steps,
                                      callbacks=callbacks_list)

# summarize history for accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
