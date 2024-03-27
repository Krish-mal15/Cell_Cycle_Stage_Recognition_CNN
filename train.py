import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras.preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
#gitt

train_path = 'archive (1)/cellData/data/train'
val_path = 'archive (1)/cellData/data/validation'

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(66,66,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4,activation='sigmoid'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam',metrics=['accuracy'])

model.summary()

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

test_dataset = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(66,66),
    batch_size=32,
    class_mode='categorical'
)

print(train_generator.class_indices)


validation_generator = test_dataset.flow_from_directory(
    val_path,
    target_size=(66,66),
    batch_size=32,
    class_mode='categorical'
)

model.fit(train_generator, epochs=7, validation_data=validation_generator)

model.save('cell_cycle_detection.keras')

