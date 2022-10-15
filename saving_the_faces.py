import cv2
import glob

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml') # Holt die Daten von cv2, mit denen man die Gesicher erkennen kann
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml') # Holt die Daten von cv2, mit denen man die Augen erkennen kann
pictures = glob.glob('./Pictures/*.jpg') # Holt die Bilder (nur .jpg) aus dem Ordner "Pictures"
img_number = 1
eyes_on_face = False # Betrachtet ob ein Auge im Bereich des Gesichtes gefunden wurde True-> Gesicht wird anerkannt

for file in pictures: # Für jedes Bild

    img = cv2.imread(file, 1) # Das Bild wird eingelesen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Macht die Bilder schwarz weiß für die KI Gesichtserkennung
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Erkennt vorhandene Gesichter in dem Bild
    # small_img = cv2.resize(img, (128, 128)) # Skalliert das Bild auf 128 x 128 

    for (x, y, w, h) in faces:
        # cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        face = img[y:y + h, x:x + w] # Wenn es ein Gesicht gibt, wird das aus dem ursprünglichen Bild raus geschnitten
        normiert = cv2.resize (face, (128, 128)) # Dann wird es auf 128² zugeschnitten
        roi_gray = gray[y:y+h, x:x+w] # Betrachtet nur den Bereich, wo das Gesicht erkannt wurde
        roi_color = img[y:y+h, x:x+w] # Betrachtet nur den Bereich, wo das Gesicht erkannt wurde
        eyes = eye_cascade.detectMultiScale(roi_gray) # Erkennt die Augen in dem Bereich de Gesichtes

        for (ex,ey,ew,eh) in eyes:
            # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2) # Macht ein Rechteck um das erkannte Auge
            # print(x, ex, y, ey, w, ew, h, eh) # Gibt die Werte aus, wo das Gesicht und das Auge gefunden wurden
            eyes_on_face = True

        if eyes_on_face: # Wenn das Gesicht anerkannt wird zeigt er das Gesicht
            # cv2.imshow(f"face: {img_number}", normiert)
            # cv2.waitKey(0) # Wartet darauf, dass das Fenster geschlossen wird
            cv2.imwrite("Faces/" + str(img_number) + ".jpg " , normiert) # Dann wird das Gesicht gespeichert
            eyes_on_face = False # Setzt die Variable wieder auf False, damit beim nächsten erkannten Gesicht wieder überprüft wird, ob das Gesicht anerkannt werden soll
            img_number = img_number + 1

    # cv2.imshow(f"img original size", img) # Zeigt das originale Bild mit den Rechtecken um die Gesichter und Augen, die erkannt wurden
    # cv2.imshow(f"img small size", small_img) # Zeigt eine kleinere Version des Originalbildes
    # cv2.waitKey(0) # Wartet darauf, dass das Fenster geschlossen wird