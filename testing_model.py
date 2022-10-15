import pickle

face = 1 #muss ein Bild von einem Gesicht sein
age = 1 #muss das Alter der Person beinhalten
gender = 1 #muss das Geschlecht der Person beinhalten

#loads the models
model_age = pickle.load(open('model_age.pkl', 'rb'))
model_gender = pickle.load(open('model_gender.pkl', 'rb'))

#predicts the age and gender
model_age.predict(face)
model_gender.predict(face)