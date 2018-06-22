#Take My Picture!
#Takes a picture with the Picam and adds it to the login for FaceID

import cv2
import numpy

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080

PICTURE_WIDTH = 512
PICTURE_HEIGHT = 512

CV_WINDOW_NAME = "Take My Picture"

def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('y')) or (ascii_code == ord('Y'))):
        return False

    return True

def main():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    
    if((camera == None) or (not camera.isOpened())):
        print("No Camera has been detected!")
        return
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    
    name = input("Please enter your name: ")
    
    cv2.namedWindow(CV_WINDOW_NAME)
    goodPicture = False
    print("Press y when you are ready for the camera to take your picture.")
    print("Try to make your face take up as much of the picture as possible.")
    while(not goodPicture):
        det, image = camera.read()
        if (not det):
            print("No Image from camera")
            break
        
        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if(prop_val < 0.0):
            print("Closed")
            break
        cv2.imshow(CV_WINDOW_NAME, image)
        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                if(len(faces) == 1):

                    for(a,b,c,d) in faces:
                        x = a
                        y = b
                        w = c
                        h = d
                    cropped_image = image[y:y+h, x:x+w]
                    print('user pressed Y')
                    while(not goodPicture):
                        cv2.imshow(CV_WINDOW_NAME, cropped_image)
                        next_raw_key = cv2.waitKey(1)
                        if(next_raw_key != -1):
                            if (handle_keys(next_raw_key) == False):
                                goodPicture = True;
                                cv2.imwrite("validated_images/" + name + ".jpg", cropped_image)
                            else:
                                break
                elif(len(faces) > 1):
                    print("Too many faces detected. Try again")
                else:
                    print("No faces detected. Try again")

    cv2.destroyAllWindows()
    camera.release()


if __name__ == "__main__":
    main()