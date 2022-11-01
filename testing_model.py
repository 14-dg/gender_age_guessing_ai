import load_faces as load_f

import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow.python.keras.models as models

#testing the age model
imgs, ages = load_f.get_pictures_age(test_train="Training")
img, age = imgs[0], ages[0]

#loads the models
model_age = models.load_model('saved_models/model_age.h5')
model_gender = models.load_model('saved_models/model_gender.h5')

#predicts the age and gender
age_predicted = model_age.predict(imgs)[0]  # type: ignore
gender_predicted = model_gender.predict(imgs)[0]  # type: ignore

number_to_gender = {0:"male", 1:"female"}

cv2.imshow(f"Predicted age: {np.argmax(age_predicted)}   actual: {age}          Predicted gender: {number_to_gender[np.argmax(gender_predicted)]}   actual: ", img)
cv2.waitKey(0)