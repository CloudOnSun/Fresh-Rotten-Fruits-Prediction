import cv2
from PIL.Image import Image


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


import os, sys

paths = [('D:\Facultate\Faculta\An_3_sem_2\dataset\Train\\freshfruits'),
         ('D:\Facultate\Faculta\An_3_sem_2\dataset\Train\\rottenfruits'),
         ('D:\Facultate\Faculta\An_3_sem_2\dataset\Test\\freshfruits'),
         ('D:\Facultate\Faculta\An_3_sem_2\dataset\Test\\rottenfruits')]
new_paths = [('D:\Facultate\Faculta\An_3_sem_2\dataset_resized\Train\\freshfruits'),
             ('D:\Facultate\Faculta\An_3_sem_2\dataset_resized\Train\\rottenfruits'),
             ('D:\Facultate\Faculta\An_3_sem_2\dataset_resized\Test\\freshfruits'),
             ('D:\Facultate\Faculta\An_3_sem_2\dataset_resized\Test\\rottenfruits')]
for i in range(0, 4):
    for item in os.listdir(paths[i]):
        im = cv2.imread(paths[i] + "\\" + item)
        f, e = os.path.splitext(item)
        imResize = image_resize(im, height=64, width=64)
        cv2.imwrite(new_paths[i] + "\\" + f + '_resized.jpg', imResize)
    print("Gata " + str(i))
