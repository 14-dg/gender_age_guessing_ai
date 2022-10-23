import glob
import os
import itertools
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
    returns pictures, genders
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
    returns pictures, ages
    """
    def get_pictures_certain_ages(ages) -> tuple[list, list]:
        """returns all pictures from all ages and the corresponding ages
        """
        ages_pictures = []
        pictures = []
        for age in ages:
            # appends the images from the folders
            # folder Age/0 -> all files  in there
            pictures.append(get_imgs_from_filenames(glob.glob(f'./Faces/Training/Age/{age}/*.jpg')))
            #appends the corresponding ages
            ages_pictures.extend([age]*len(pictures[-1]))
        # flattens the list of pictures, so they are in one big one dimensional array 
        return list(itertools.chain(*pictures)), ages_pictures

    def get_available_ages() -> list:
        path = "./Faces/Training/Age/"    
        return [int(file) for file in os.listdir(path)]
    
    ages = get_available_ages()
    pictures, ages_pictures = get_pictures_certain_ages(ages)
       
    return pictures, ages_pictures

if __name__ == "__main__":
    #imgs, genders = get_pictures_gender()
    imgs, ages = get_pictures_age()
    for ind, img in enumerate(imgs):
        cv2.imshow(f"The person is {ages[ind]}", img)
        cv2.waitKey(0)