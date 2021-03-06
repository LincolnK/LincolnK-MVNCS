#Video Viewer

import cv2
import numpy
import time

from PIL import Image, ImageDraw
from google.cloud import vision
from google.cloud.vision import types

CV_WINDOW_NAME = "Simple Video Viewer"
FILENAME = "5ft_test.mp4"

def detect_face(image, max_results=4):
    client = vision.ImageAnnotatorClient()
    content = image
    image = types.Image(content=content)
    return client.face_detection(image=image).face_annotations


def highlight_faces(image, faces, output_filename):
    for face in faces:
        box = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
        coords = []
        for vertex in face.bounding_poly.vertices:
            coords.append((vertex.x,vertex.y))
        x = coords[0][0]
        y = coords[0][1]
        w = coords[1][0]-x
        h = coords[2][1]-y
        if(w < 0):
            x = x + w
            w = -1 * w   
        if(h < 0):
            y = y + h
            h = -1 * h
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
    cv2.imwrite(output_filename, image)
    return



def main():
    filename = input("Plese Enter the file you would like to watch: ")
    video = cv2.VideoCapture(filename)
    if(not video.isOpened()):
        print("Failed to open the video file: " + filename)
        return
    frames = 0
    cv2.namedWindow(CV_WINDOW_NAME)
    startTime = time.time()
    set = True
    while(True):
        return_val, video_image = video.read()
        if (not return_val):
            print("No Image from camera")
            break
        p_img = cv2.imencode('.jpg', video_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])[1].tostring()
        if(set):
            x = detect_face(p_img,4)
            print('Found {} face{}'.format(len(x), '' if len(x) == 1 else 's'))
            highlight_faces(video_image, x, "outfile.jpg")
            set = False
        cv2.imshow(CV_WINDOW_NAME, video_image)
        cv2.waitKey(1)
        frames +=1
        
    endTime = time.time()
    totalTime = endTime - startTime
    print("FPS: " + str(frames / totalTime))
    cv2.destroyAllWindows()
    video.release()
    print("done")

if __name__ == "__main__":
    main()