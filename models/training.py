import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.models import Sequential
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

data_path = 'data/train/'
valid_path = 'data/validation'

train_data = ImageDataGenerator(rescale=1.0/255.0, shear_range=0.2,
                                zoom_range=0.2, horizontal_flip=True)

generator = train_data.flow_from_directory(data_path,
                                           target_size=(224, 224),
                                           color_mode='rgb',
                                           batch_size=32,
                                           class_mode='categorical',
                                           shuffle=True)

valid_generator = train_data.flow_from_directory(valid_path,
                                           target_size=(224, 224),
                                           color_mode='rgb',
                                           batch_size=32,
                                           class_mode='categorical',
                                           shuffle=True)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dense(units=2, activation='sigmoid'))

model.summary()

model.compile(optimizer='Adam', loss='binary_crossentropy',
                 metrics=['accuracy'])

step_size = generator.n // generator.batch_size
valid_step_size = valid_generator.n // valid_generator.batch_size

hist = model.fit_generator(generator,
                              steps_per_epoch=step_size,
                              epochs=50,
                              validation_data=valid_generator,
                              validation_steps=valid_step_size)

accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']
loss = hist.history['loss']
val_loss = hist.history['val_loss']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'r', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation Accuracy')
plt.title('Training and Validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

model.save('/kaggle/working/model.h5')