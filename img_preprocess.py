import cv2
from matplotlib import pyplot as plt
import numpy as np
import time as t


def crop_image(image, size):
    image_list = []

    #Find height, width of input image
    height = image.shape[0]
    width = image.shape[1]

    #Find out how many slices we can do for height and width based on size
    height_slices = int(height / size)
    width_slices = int(width / size)

    #Set start and stop parameters for image cropping
    h_start, h_stop = 0, size
    w_start, w_stop = 0, size

    #This for loop will
    for i in range(height_slices):
        for j in range(width_slices):
            crop_img = image[h_start:h_stop, w_start:w_stop]
            image_list.append(crop_img)
            w_start += size
            w_stop += size
        h_start += size
        h_stop += size
        w_start = 0
        w_stop = size

    return image_list

def rotate_images_4x(image_list):
    new_image_list = []

    #Takes the image from the list, converts to NP array, flips it 90 degrees
    #and saves it for 4 total images
    for i in image_list:
        img = np.array(i)
        new_image_list.append(img)
        for j in range(3):
            img = np.rot90(img)
            new_image_list.append(img)
    return new_image_list

def mirror_images(image_list):
    new_image_list = []
    for i in image_list:
        img = np.array(i)
        new_image_list.append(img)
        img = np.fliplr(img)
        new_image_list.append(img)
    return new_image_list


def save_images(image_list):
    for i in range(len(image_list)):
        cv2.imwrite('test_images/img_{}.jpg'.format(i), image_list[i])

if __name__ == '__main__':
    PATH = "maroon_bells.jpg"
    image = cv2.imread(PATH)

    #change the colors from BGR to RGB - this is unnecessary because imwrite flips it back
    # BGRflags = [flag for flag in dir(cv2) if flag.startswith('COLOR_BGR') ]
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print("Height:\t\t%i pixels\nWidth:\t\t%i pixels\nChannels:\t%i" % image.shape)
    print("pixel at (0,0) [B,G,R]:\t[%i,%i,%i]" % tuple(image[0,0,:]))
    print("data-type: %s " % image.dtype)

    #This is how to resize the image
    resized_image = cv2.resize(image, (100, 50))
    # plt.imshow(resized_image)
    # plt.show()

    #Pipeline Test
    l1 = []
    l1.append(image)
    rotated = rotate_images_4x(l1)
    mirrored_rotated = mirror_images(rotated)
