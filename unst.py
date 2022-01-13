from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
import cv2
import os

imgin = r"data/ImageIn/"
imgout = r"data/ImageOut/"
currentFrame = 0
count = 0

# Use OpenCV to pull all frames from video source
cap = cv2.VideoCapture('data/VideoIn/example.mp4')
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is False:
        break
    name = imgin + str(currentFrame) + '.jpg'
    print('Creating frame: ' + name)
    cv2.imwrite(name, frame)
    currentFrame += 1
cap.release()
cv2.destroyAllWindows()

# Creates array from OpenCV frames
for entry in os.scandir(imgin):
    im = Image.open(imgin + str(count) + ".jpg")
    im_array = np.array(im)
    # print(im_array)
    # print(im_array.shape)
    # print(type(im))
    # im = Image.save(imgout + str(count) + ".jpg")
    output = Image.fromarray(im_array)
    # plt.Image.save(imgout + str(count) + ".jpg", im_array)
    plt.imshow(output)
    # plt.show()
    plt.savefig(imgout + str(count))
    count += 1
