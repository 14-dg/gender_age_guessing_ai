import tensorflow as tf
import pickle

faces = 1 # Müssen alle Gesichter als 128² Schwarzweiß Bild sein
ages = 1 # Muss das Alter der Menschen haben

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