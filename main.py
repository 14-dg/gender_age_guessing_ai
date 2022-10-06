import cv2
import os
import tensorflow
import dlib

img = cv2.imread(os.path.join('Assets', 'test.jpg'))
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_cascade = dlib.get_frontal_face_detector()

#face_cascadecv2 = cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')
#facescv2 = face_cascadecv2.detectMultiScale(grayimg, )





def save_img(img, name, bbox, width = 144, height = 144):
    x, y, w, h, = bbox
    image = img[y: h, x: w]
    image = cv2.resize(image, (width, height))
    cv2.imwrite(name + ".jpg", image)

def rec_faces():
    faces = face_cascade(grayimg)
    for counter, face in enumerate(faces):
        x1, x2, y1, y2 = face