#UDP Face Detection Local
# UDP Face Watch Video

import csv
import socket
import numpy as np
import cv2
import math
import time
import sys

CV_WINDOW_NAME = "Lincoln's UDP Video Streamer Client"

DEBUG = False
    
    
def runTest(width, height, filename, csvw):
    if(DEBUG):
        print("Starting " + filename)
    video = cv2.VideoCapture(filename)
    if(not video.isOpened()):
        print("Failed to open the video file: " + filename)
        return
    detectedFrames = 0
    totalFrames = 0
    
    timerA = None
    timerB = None
    timerC = None
    timerD = None
    timerE = None
    
    readTime = []
    cascadeTime = []
    showTime = []
    cycleTime = []
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    cv2.namedWindow(CV_WINDOW_NAME)
    while(True):
        set = True
        timerA = time.time()
        return_val, video_image = video.read()
        if (not return_val):
            print("No Image from camera")
            break
        video_image = cv2.resize(video_image, (width, height))
        
        
        timerB = time.time()
        gray = cv2.cvtColor(video_image, cv2.COLOR_BGR2GRAY)
        print(type(gray))
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        
        timerC = time.time()
        for(x,y,w,h) in faces:
            if(set):
                detectedFrames += 1
                set = False
            cv2.rectangle(video_image,(x,y),(x+w,y+h),(255,0,0),2)
        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if(prop_val < 0.0):
            print("Closed")
            break
        totalFrames += 1
        cv2.imshow(CV_WINDOW_NAME, video_image)
        cv2.waitKey(1)
        
        
        timerD = time.time()
        readTime.append(timerB-timerA)
        cascadeTime.append(timerC-timerB)
        showTime.append(timerD-timerC)
        cycleTime.append(timerD-timerA)
        
    avg_readTime = str(sum(readTime)/float(len(readTime)))
    avg_cascadeTime = str(sum(cascadeTime)/float(len(cascadeTime)))
    avg_showTime = str(sum(showTime)/float(len(showTime)))
    avg_cycleTime = str(sum(cycleTime)/float(len(cycleTime)))
    avg_framerate = str(1/(sum(cycleTime)/float(len(cycleTime))))
    
    csvw.writerow([str(width) + "x" + str(height) + ";" + filename + ";" + str(totalFrames) + ";" + str(detectedFrames) + ";" + avg_readTime + ";" + avg_cascadeTime + ";" + avg_showTime + ';' + avg_cycleTime + ';' + avg_framerate])
    
    video.release()
    return
            
    
    
def main():
    with open('Results/LocalTest.csv', 'w', newline='') as csvfile:
        csvw = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        print('You ahve 10 seconds to unplug everything')
        time.sleep(10)
        print('Starting...')
        csvw.writerow(["Resolution;Test;Total Frames;Detected Frames;Average Read Time;Average Cascade Time;Average Show Time;Average Cycle Time;Average Framerate;"])
        try:
            
            '''runTest(640, 480, '3ft_test.mp4', csvw)
            runTest(1280, 720, '3ft_test.mp4', csvw)
            runTest(1920, 1080, '3ft_test.mp4', csvw)'''
            
            runTest(640, 480, '5ft_test.mp4', csvw)
            time.sleep(5)
            runTest(1280, 720, '5ft_test.mp4', csvw)
            time.sleep(5)
            runTest(1920, 1080, '5ft_test.mp4', csvw)
            
            '''csvw.writerow(["NA"])
            runTest(1280, 720, '10ft_test.mp4', csvw)
            runTest(1920, 1080, '10ft_test.mp4', csvw)
            
            csvw.writerow(["NA"])
            runTest(1280, 720, '15ft_test.mp4', csvw)
            runTest(1920, 1080, '15ft_test.mp4', csvw)
            
            csvw.writerow(["NA"])
            runTest(1280, 720, '20ft_test.mp4', csvw)
            runTest(1920, 1080, '20ft_test.mp4', csvw)'''
        except:
            cv2.destroyAllWindows()
            print('Error')
            raise
    
    cv2.destroyAllWindows()
    print('Done')
    
if __name__ == "__main__":
    main()