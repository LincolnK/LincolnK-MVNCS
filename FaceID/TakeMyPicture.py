#Take My Picture!
#Takes a picture with the Picam and adds it to the login for FaceID

import cv2
import numpy

RESOLUTION_WIDTH = 512  #1280
RESOLUTION_HEIGHT = 640 #1024

CV_WINDOW_NAME = "Take My Picture"

def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('p')) or (ascii_code == ord('P'))):
        return False

    return True

def whiten_image(image):
    mean = numpy.mean(image)
    std = numpy.std(image)
    std_adjusted = numpy.maximum(std, 1.0 / numpy.sqrt(image.size))
    whitened_image = numpy.multiply(numpy.subtract(image, mean), 1 / std_adjusted)
    return whitened_image

def process_image(image):
    preprocessed_image = cv2.resize(image, (RESOLUTION_WIDTH, RESOLUTION_HEIGHT))
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    preprocessed_image = whiten_image(preprocessed_image)
    return preprocessed_image

def main():
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
    
    if((camera == None) or (not camera.isOpened())):
        print("No Camera has been detected!")
        return
    
    name = input("Please enter your name: ")
    
    cv2.namedWindow(CV_WINDOW_NAME)
    print("Press p when you are ready for the camera to take your picture.")
    print("Try to make your face take up as much of the picture as possible.")
    while True:
        det, image = camera.read()
        if (not det):
            print("No Image from camera")
            break
        #image = process_image(image)
        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if(prop_val < 0.0):
            print("Closed")
            break
        cv2.imshow(CV_WINDOW_NAME, image)
        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                print('user pressed P')
                cv2.imwrite("validated_images/" + name + ".jpg", image)
                cv2.destroyAllWindows()
                break
    camera.release()


if __name__ == "__main__":
    main()