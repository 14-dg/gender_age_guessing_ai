import cv2
import glob

face_cascade = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml') # Holt die Daten von cv2, mit denen man die Gesicher erkennen kann
pictures = glob.glob('./Pictures/*.jpg') # Holt die Bilder (nur .jpg) aus dem Ordner "Pictures"
for file in pictures[]: # Für jeder Bild
    img = cv2.imread(file, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    try :
        for ( x , y , w , h ) in faces :
            roi_color = img [ y : y + h , x : x + w ]
        normiert = cv2.resize (roi_color, (128,128))
        cv2.imwrite (" extracted_faces / " + str (img_number) + ".jpg " , normiert)
    except :
        print ("Kein Gesicht gefunden")
    img_number = img_number + 1