import tensorflow as tf
import pickle

faces = 1 # Müssen alle Gesichter als 128² Schwarzweiß Bild sein
genders = 1 # Muss das Geschlecht der Person als 0 oder 1 haben

model_gender = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(30, (3, 3), activation = 'relu', input_shape = (128, 128, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(30, (3, 3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(512, activation = 'relu'),
    tf.keras.layers.Dense(2, activation = 'softmax')])
model_gender.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
model_gender.fit(faces, genders, epochs = 50)

pickle.dump(model_gender, open('model_gender.pkl', 'wb'))