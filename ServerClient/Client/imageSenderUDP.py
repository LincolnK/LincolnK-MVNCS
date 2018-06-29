# Image Sender TCP

import socket
import numpy as np
import cv2
import math


USENET = True

UDP_IP = "131.230.191.195"
UDP_PORT = 5005
BUFFER_SIZE = 1024

def main():
    print("start")
    img = cv2.imread("picture.jpg")
    p_img = img.dumps()
    length = len(p_img)
    
    num_packets = str(math.ceil((len(p_img)/BUFFER_SIZE)))+"|"
    print(num_packets)

    
    if(USENET):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect((UDP_IP, UDP_PORT))
        sock.send(num_packets.encode())
        for i in range(int(len(p_img)/BUFFER_SIZE)+1):
            #print(i)
            sock.send(p_img[i*BUFFER_SIZE:(i+1)*BUFFER_SIZE])
        sock.close()
    print("stop")

    

if __name__ == "__main__":
    main()