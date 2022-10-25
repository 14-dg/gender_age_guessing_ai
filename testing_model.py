import pickle
import load_faces as load_f

#testing the age model
faces, ages = load_f.get_pictures_age(test_train="Testing")
face, age = faces[0], ages[0]

#loads the models
model_age = pickle.load(open('model_age.pkl', 'rb'))
model_gender = pickle.load(open('model_gender.pkl', 'rb'))

#predicts the age and gender
model_age.predict(face)
model_gender.predict(face)