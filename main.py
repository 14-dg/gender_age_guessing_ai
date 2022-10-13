import cv2
import glob

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Holt die Daten von cv2, mit denen man die Gesicher erkennen kann
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
pictures = glob.glob('./Pictures/*.jpg') # Holt die Bilder (nur .jpg) aus dem Ordner "Pictures"

eyes_on_face = False
for file in pictures: # Für jedes Bild
    img_number = 1
    img = cv2.imread(file, 1) # Das Bild wird eingelesen
    small_img = cv2.resize(img, (0,0), fx=0.3, fy=0.3) #macht das Bild 0.3 mal so groß
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Macht die Bilder schwarz weiß für die KI Gesichtserkennung
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Erkennt vorhandene Gesichter in dem Bild
    try :
        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            face = img[y:y + h, x:x + w] # Wenn es ein Gesicht gibt, wird das aus dem ursprünglichen Bild raus geschnitten
            normiert = cv2.resize (face, (128,128)) # Dann wird es auf 128² zugeschnitten (anscheinend später für die KI relevant)
            
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex,ey,ew,eh) in eyes:
                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
                print(x, ex, y, ey, w, ew, h, eh)
                eyes_on_face = True
            
            if eyes_on_face:
                cv2.imshow(f"face: {img_number}", normiert)
                cv2.waitKey(0)
                
                cv2.imwrite ("Faces/" + str(img_number) + ".jpg " , normiert) # Dann wird das Gesicht gespeichert
                img_number = img_number + 1
                
                eyes_on_face = False
                
        cv2.imshow(f"img original size", img)
        cv2.imshow(f"img small size", small_img)
        cv2.waitKey(0)
            
    except :
        print ("Kein Gesicht gefunden") # Falls es kein Gesicht gibt, wird die Nachricht gedruckt