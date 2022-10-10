import cv2
import glob

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Holt die Daten von cv2, mit denen man die Gesicher erkennen kann
pictures = glob.glob('./Pictures/*.jpg') # Holt die Bilder (nur .jpg) aus dem Ordner "Pictures"
for file in pictures: # Für jedes Bild
    img = cv2.imread(file, 1) # Das Bild wird eingelesen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Macht die Bilder schwarz weiß für die KI Gesichtserkennung
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Erkennt vorhandene Gesichter in dem Bild
    try :
        for (x, y, w, h) in faces:
            face = img[y:y + h, x:x + w] # Wenn es ein Gesicht gibt, wird das aus dem ursprünglichen Bild raus geschnitten
        normiert = cv2.resize (face, (128,128)) # Dann wird es auf 128² zugeschnitten (anscheinend später für die KI relevant)
        cv2.imwrite ("Faces/" + str(img_number) + ".jpg " , normiert) # Dann wird das Gesicht gespeichert
    except :
        print ("Kein Gesicht gefunden") # Falls es kein Gesicht gibt, wird die Nachricht gedruckt
    img_number = img_number + 1