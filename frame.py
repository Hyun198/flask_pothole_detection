import cv2
import os

#file_path = 'C:/Users/user-pc/Desktop/visual/project/static/video'

def frame(path):
    os.chdir(path)
    file_names = os.listdir()
    for filename in file_names:
        if os.path.splitext(filename)[1] == '.MP4':
            file_path1 = filename

    filePath = os.path.join(path, file_path1)  # 파일명만 받아올 수 있으면 됨
    filePath = filePath.replace("\\", "/")

    outpath='C:/Users/user-pc/Desktop/visual/project/input/frames/'
    vidcap = cv2.VideoCapture(filePath)
    def getFrame(sec):
        vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        hasFrames,image = vidcap.read()
        if hasFrames:
            cv2.imwrite(outpath+"image"+str(count)+".jpg", image)     # save frame as JPG file
        return hasFrames
    sec = 0
    frameRate = 0.5 #//it will capture image in each 0.5 second
    count=1
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)





