from net.build import TFNet
from utils import loader
import cv2
import glob
import time
import numpy as np
from matplotlib import pyplot as plt
import sys

options = {"test": "test/", "model": "cfg/tiny-yolo-udacity.cfg", "backup": "ckpt/","load": 8987, "gpu": 1.0}

tfnet = TFNet(options)


####################################
#read a video input file
print("start to reading mp4 frames")
if (sys.argv[1] is not NONE):
    vin = sys.argv[1] #'MOVI0019_1min32secto3min2sec.mp4'
    vout = '_'.join(['YOLO', vin])
    fname = '/'.join(['./test_videos', vin])
    print('video in file: {}'.format(vin))
    print('video out file: {}'.format(vout))

    inFile = cv2.VideoCapture(vin) #'./test_videos/MOVI0019_1min32secto3min2sec.mp4')

    #check if the input file opened successfully
    if (inFile.isOpened() == False):
        print("Error opening video stream on file")

    #define the codec and create videowriter object
    fps = 20
    frame_size = (int(inFile.get(3)), int(inFile.get(4)))    #tuple(result.shape[1::-1])
    print("frame_size: {}".format(frame_size))
    #writer = cv2.VideoWriter("./test_videos/YOLO_MOVI0019_1min32secto3min2sec.mp4", cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size, True)  
     writer = cv2.VideoWriter(vout,
         cv2.VideoWriter_fourcc(*'MP4V'), fps, frame_size, True)  

    #read until video is completed
    while(inFile.isOpened()):
        #Capture frame by frame
        ret, frame = inFile.read()

        if ret == True:
            #display frame
            #plt.imshow(frame)
            #plt.show()
            result, boxInfo = tfnet.return_predict(frame)
            #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
            #plt.imshow(result)
            #plt.show()
            writer.write(result)
        else:
            #if no frame break while loop
            writer.release() 
            print("end of mp4 video file conversion") 
            break
#####################################
else: #if just for testing purpose, not video purpose
    images = glob.glob('./test/*.jpg')
    i = 0
    average = []
    for image in images:
        t = time.time()
        imgcv = cv2.imread(image)
        name = image
        result, boxInfo = tfnet.return_predict(imgcv,name)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()
        t2 = time.time()
        average.append(t2-t)
        i += 1
        print(i,'time1:',(t2-t))
    final = np.mean(average)
    print(i,'images processed in avg: ', round(final,6), 'seconds per image')
