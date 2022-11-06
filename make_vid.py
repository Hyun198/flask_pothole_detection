import cv2, os
import numpy as np
import glob
from os.path import isfile, join


def make_vid():
    img_array =[]
    fps = 8
    for filename in glob.glob('C:/Users/user-pc/Desktop/visual/project/input/frames/*.jpg'):
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)

    out = cv2.VideoWriter('static/result_vid/project.mp4',cv2.VideoWriter_fourcc(*'h264'),fps,size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

if __name__ == '__main__':
    make_vid()