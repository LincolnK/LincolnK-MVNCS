# UDP Face Watch Video

import csv
import socket
import numpy as np
import cv2
import math
import time
import sys


RESOLUTION_WIDTH = 1920
RESOLUTION_HEIGHT = 1080
CV_WINDOW_NAME = "Lincoln's UDP Video Streamer Client"


UDP_IP = "35.237.179.67"
#UDP_IP = "131.230.191.195"
UDP_PORT = 5052
BUFFER_SIZE = 4096

SLEEP_TIME = 0.000 #0.00035

START_DELAY = 0



DEBUG = False



def parseCoords(input):
    firstParse = input.split("<")
    faces = []
    for substr in firstParse[1:]:
        face_coord = substr.split("|")
        face = []
        for val in face_coord:
            face.append(int(val))
        faces.append(tuple(face))
    return faces


def sendImg(sock, image):
    p_img = cv2.imencode('.jpg', image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tostring()
    length = len(p_img)
    if(DEBUG):
        print("Attempting to send Image of size " + str(length))
    num_packets = "LEN*" + str(math.ceil((len(p_img)/BUFFER_SIZE)))
    if(DEBUG):
        print("-------------------------------")
        print("Packet: " + num_packets)
        print("Length: " + str(len(p_img)))
        print("Size:   " + str(sys.getsizeof(p_img)))
    try:
        sock.send(num_packets.encode('utf-8'))
        for i in range(int(len(p_img)/BUFFER_SIZE)+1):
            time.sleep(SLEEP_TIME)
            sock.send(p_img[i*BUFFER_SIZE:(i+1)*BUFFER_SIZE])
        if(DEBUG):
            print("Image Sent")
        return
    except:
        print("Send Failure")
    
    
def getReply(sock):
    try:
        if(DEBUG):
            print("Awaiting Reply")
        sock.settimeout(0.5)
        reply, addr = sock.recvfrom(BUFFER_SIZE)
        decoded_reply = reply.decode('utf-8')
        if(decoded_reply[:4] == "ERR*"):
            if(DEBUG):
                print("Bad Reply recieved")
            return None
        else:
            faces = parseCoords(decoded_reply)
            if(DEBUG):
                print("Reply recieved")
            return faces
    except:
        if(DEBUG):
            print("No Response")
    
    
    
def terminate(sock):
    try:
        sock.send('TER*'.encode('utf-8'))
    except:
        print("Send Failure")
    
    
    
def runTest(sock, width, height, filename, csvw):
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
    sendTime = []
    rcvTime = []
    showTime = []
    cycleTime = []

    
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
        greyImg = cv2.cvtColor(video_image, cv2.COLOR_BGR2GRAY)
        sendImg(sock, greyImg)
        
        
        timerC = time.time()
        faces = getReply(sock)
        
        
        timerD = time.time()
        if(faces is not None):
            for(x,y,w) in faces:
                if(set):
                    set = False
                    detectedFrames += 1
                cv2.rectangle(video_image,(x,y),(x+w,y+w),(255,0,0),2)
                
        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if(prop_val < 0.0):
            print("Closed")
            break
        totalFrames += 1
        cv2.imshow(CV_WINDOW_NAME, video_image)
        cv2.waitKey(1)
        timerE = time.time()
        readTime.append(timerB-timerA)
        sendTime.append(timerC-timerB)
        rcvTime.append(timerD-timerC)
        showTime.append(timerE-timerD)
        cycleTime.append(timerE-timerA)
        
    avg_readTime = str(sum(readTime)/float(len(readTime)))
    avg_sendTime = str(sum(sendTime)/float(len(sendTime)))
    avg_rcvTime = str(sum(rcvTime)/float(len(rcvTime)))
    avg_showTime = str(sum(showTime)/float(len(showTime)))
    avg_cycleTime = str(sum(cycleTime)/float(len(cycleTime)))
    avg_framerate = str(1/(sum(cycleTime)/float(len(cycleTime))))
    
    csvw.writerow([str(width) + "x" + str(height) + ";" + filename + ";" + str(totalFrames) + ";" + str(detectedFrames) +";" + avg_readTime + ";" + avg_sendTime + ";" + avg_rcvTime + ";" + avg_showTime + ';' + avg_cycleTime + ';' + avg_framerate])
    
    video.release()
    return
            
    
    
def main():
    print('You have ' + str(START_DELAY) + ' seconds to unplug everything')
    time.sleep(START_DELAY)
    print('starting')
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect((UDP_IP, UDP_PORT))
    try:
        with open('Results/GoogleUsingEthernet.csv', 'w', newline='') as csvfile:
            csvw = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csvw.writerow(["Resolution;Test;Total Frames;Detected Frames;Average Read Time;Average Send Time;Average Recieve Time;Average Show Time;Average Cycle Time;Average Framerate"])
           
            runTest(sock, 640, 480, '3ft_test.mp4', csvw)
            runTest(sock, 1280, 720, '3ft_test.mp4', csvw)
            runTest(sock, 1920, 1080, '3ft_test.mp4', csvw)
            
            runTest(sock, 640, 480, '5ft_test.mp4', csvw)
            runTest(sock, 1280, 720, '5ft_test.mp4', csvw)
            runTest(sock, 1920, 1080, '5ft_test.mp4', csvw)
            
            csvw.writerow(["NA"])
            runTest(sock, 1280, 720, '10ft_test.mp4', csvw)
            runTest(sock, 1920, 1080, '10ft_test.mp4', csvw)
            
            csvw.writerow(["NA"])
            runTest(sock, 1280, 720, '15ft_test.mp4', csvw)
            runTest(sock, 1920, 1080, '15ft_test.mp4', csvw)
            
            csvw.writerow(["NA"])
            runTest(sock, 1280, 720, '20ft_test.mp4', csvw)
            runTest(sock, 1920, 1080, '20ft_test.mp4', csvw)
            
    except:
        terminate(sock)
        sock.close()
        cv2.destroyAllWindows()
        print('Error')
        raise
    terminate(sock)
    sock.close()
    cv2.destroyAllWindows()
    print('Done')
    
if __name__ == "__main__":
    main()