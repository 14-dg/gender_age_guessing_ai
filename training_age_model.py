import tensorflow as tf
import pickle

import load_faces as load_f_t

faces = load_f_t.get_pictures_age()
ages = 1

model_age = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(30, (3, 3), activation = 'relu', input_shape = (128, 128, 1)),
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