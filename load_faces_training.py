import glob
import cv2


def get_imgs_from_filenames(pictures)->list:
    """loads the imgs after knowing where they are located
    """
    imgs = []
    for file in pictures:
        img = cv2.resize(cv2.imread(file, 1), (128, 128))
        imgs.append(img)
    return imgs


def get_pictures_gender()-> tuple[list, list]:
    """
    return imgs in array form
    male -> 1 female -> 0
    """
    def get_pictures_gender_male()-> tuple[list, list]:
        pictures = glob.glob('./Faces/Training/Gender/male/*.jpg')
        genders = [1]*len(pictures)
        return get_imgs_from_filenames(pictures), genders

    def get_pictures_gender_female()-> tuple[list, list]:
        pictures = glob.glob('./Faces/Training/Gender/female/*.jpg')
        genders = [0]*len(pictures)
        return get_imgs_from_filenames(pictures), genders
    pictures_male, genders_male = get_pictures_gender_male()
    pictures_female, genders_female = get_pictures_gender_female()
    return pictures_male+pictures_female, genders_male+genders_female

def get_pictures_age()-> tuple[list, list]:
    """
    return imgs in array form
    ages -> represent the age of the person
    """
    pictures = glob.glob('./Faces/Training/Age*.jpg')
    ages = []
    return get_imgs_from_filenames(pictures), ages

if __name__ == "__main__":
    imgs, genders = get_pictures_gender()
    for ind, img in enumerate(imgs):
        cv2.imshow(f"The person is a {genders[ind]}", img)
        cv2.waitKey(0)