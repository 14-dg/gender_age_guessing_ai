import pickle
import numpy as np

import load_faces as load_f

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

faces, ages = load_f.get_pictures_age(test_train="Training")

model_age = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(30, (3, 3), activation = 'relu', input_shape = (128, 128, 3)),#CHANGED input_shape from (128, 128, 1) -> (128, 128, 3)
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(30, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(100, activation = 'softmax')])
model_age.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model_age.fit(faces, ages, epochs = 50)

pickle.dump(model_age, open('model_age.pkl', 'wb'))