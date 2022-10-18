import glob
import cv2

#loads the imgs after knowing where they are located
def get_imgs_from_filenames(pictures)->list:
    imgs = []
    for file in pictures:
        img = cv2.resize(cv2.imread(file, 1), (128, 128))
        imgs.append(img)
    return imgs

'''
return imgs in array form
to access one single img use a for loop
'''
def get_pictures_gender()->list:
    pictures = glob.glob('./Pictures/*.jpg')
    return get_imgs_from_filenames(pictures)

def get_pictures_age()->list:
    pictures = glob.glob('./Pictures/*.jpg')
    return get_imgs_from_filenames(pictures)

if __name__ == "__main__":
    imgs = get_pictures_gender()
    for img in imgs:
        print(img)
        cv2.imshow("test", img)
        cv2.waitKey(0)