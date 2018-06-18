import sys
sys.path.insert(0, "ncapi2_shim")
import mvnc_simple_api as mvnc

import numpy
import cv2
import os

validated_image_list = os.listdir("./validated_images/")
GRAPH_FILENAME = "facenet_celeb_ncs.graph"
CV_WINDOW_NAME = "Lincoln Kinley's Face ID app"

CAMERA_INDEX = 0
RESOLUTION_WIDTH = 512   # was 640, tested at 720 turned blue, tested at 1280 couldn't recognize faces
RESOLUTION_HEIGHT = 640   # was 480, tested at 1280 turned blue, tested at 720 couldn't recognize faces
NETWORK_WIDTH = 160       # was 160
NETWORK_HEIGHT = 160      # was 160

MATCH_THRESHOLD = 0.3




'''-----------------------------------------------------------------------------------------------------
This function is where the neural network is actually put into play.
Preproccess function sets up the image so that it can properly be read by the neural network.
LoadTensor is the function that simulates the Neural Network

Parameters
    image (type mat)
        This is the image that was taken by the camera, sent in whatever resolution the camera took the picture at
    graph (type graph)
        This is the .graph file that the Neural Network uses to find facial features       
'''-----------------------------------------------------------------------------------------------------
def infer(image, graph):
    resized_image = preprocess(image)
    graph.LoadTensor(resized_image.astype(numpy.float16), None)
    output, userobj = graph.GetResult()
    return output
    
    
    
    
'''-----------------------------------------------------------------------------------------------------
This funtion is used to put an overlay on the output window. This makes it extra clear if a face is detected
The first part of the function puts text onto the screen
The second part arranges a border around the window, either green or red. if matching is true, the border is green. Otherwise it is red


image (type mat)
    This is the image that was taken by the camera, sent in whatever resolution the camera took the picture at
info (type string)
    This is any info that should be listed in the upper right corner. I use it for a FPS counter, but the detected persons name could also be useful
matching (type bool)
    This is a bool statement used to determine if the face detected by the camera matches one in the verifed faces directory
'''-----------------------------------------------------------------------------------------------------
def frame_update(image, info, matching):
    if(info != None):
        cv2.putText(image, info, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    rect_width = 10
    offset = int(rect_width/2)
    if(matching):
        cv2.rectangle(image, (offset, offset), (image.shape[1]-offset-1, image.shape[0]-offset-1), (0, 255, 0), 10)
    else:
        cv2.rectangle(image, (offset, offset), (image.shape[1]-offset-1, image.shape[0]-offset-1), (0, 0, 255), 10)




'''-----------------------------------------------------------------------------------------------------
This function whitens the image to make it easier for the neural network to detect features of the face.

image (type mat)
        This is the image that was taken by the camera, sent in whatever resolution the camera took the picture at
'''-----------------------------------------------------------------------------------------------------
def whiten_image(image):
    mean = numpy.mean(image)
    std = numpy.std(image)
    std_adjusted = numpy.maximum(std, 1.0 / numpy.sqrt(image.size))
    whitened_image = numpy.multiply(numpy.subtract(image, mean), 1 / std_adjusted)
    return whitened_image




'''-----------------------------------------------------------------------------------------------------
This function does all of the editing to make the image suitable to be interpreted by the neural network
First, resize is imprortant because most neural networks can only accept a specific size image. This resize makes the image that size
Next, the format of the image is changed from BRG to RGB. This neural network wants RGB arrays, but the camera takes RGB arrays, meaning the format must be changed.
Last, whiten image whitens the image

image (type mat)
        This is the image that was taken by the camera, sent in whatever resolution the camera took the picture at
'''-----------------------------------------------------------------------------------------------------
def preprocess(image):
    preprocessed_image = cv2.resize(image, (NETWORK_WIDTH, NETWORK_HEIGHT))
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    preprocessed_image = whiten_image(preprocessed_image)
    return preprocessed_image




'''-----------------------------------------------------------------------------------------------------
This function compares two outputs of the neural network
The total difference is between 0 and 2.15 with 0 being an exact copy

face1 (type float[])
    this is the output of the neural network after it processes a face.
face2 (type float[])
    same as face1
'''-----------------------------------------------------------------------------------------------------
def face_match(face1, face2):
    if(len(face1) != len(face2)):
        print("Length Misatch")
        return false
    total_difference = 0
    for i in range(0,len(face1)):
        difference = numpy.square(face1[i] - face2[i])
        total_difference += difference
    print("Total Difference is: " + str(total_difference))
    return total_difference




'''-----------------------------------------------------------------------------------------------------
This function makes up the bulk of the program
It begins by opening the camera and a window which will show the output of the camera, the it goes into the main loop
the main loop reads the camera. If the camera gets disconected the program terminates
test_output is the output of the neural network after it tests the image taken by the camera
next the program finds the face that is most similar to the on analyzed by the neural network
The program looks for a valid face for 5 cycles. This is to filer out any noise that the neural network might have
This also helps the program if one of the images is bad, for example if one of the images comes out too blurry, the program will recognize the person after two bad framesany more bad frames and it will reset.

output (type string[])
    This is the list of all the verified faces
valid_image_name (type string)
    This is the listed directory of the verified faces
graph (type graph)
    This is the .graph that the neural network will use
'''-----------------------------------------------------------------------------------------------------
def run_camera(output, valid_image_name, graph):
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
    
    if((camera == None) or (not camera.isOpened())):
        print("No Camera has been detected!")
        return
    
    frame_count = 0
    
    cv2.namedWindow(CV_WINDOW_NAME)
    
    match=False
    rechecks = 0
    while True:
        return_val, video_image = camera.read()
        if (not return_val):
            print("No Image from camera")
            break
        test_output = infer(video_image, graph)
            
        min_distance = 100
        min_index = -1
            
        for i in range(0,len(output)):
            distance = face_match(output[i], test_output)
            if distance < min_distance:
                min_distance = distance
                min_index = i
        if(min_distance<=MATCH_THRESHOLD):
            frame_name = validated_image_list[min_index]
            if(rechecks > 5):
                print("Found! " + validated_image_list[min_index])
                found_match = True
                if(rechecks > 7):
                    rechecks = 7
            else:
                print("Checking... " + validated_image_list[min_index])
                found_match = False
            rechecks += 1
        else:
            frame_name = "No Verified Face"
            found_match = False
            print("Fail!")
            if(rechecks <= 5):
                rechecks = 0
            else:
                rechecks -= 1
        frame_update(video_image, frame_name[0:-4], found_match)

        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if(prop_val < 0.0):
            print("Closed")
            break
        cv2.imshow(CV_WINDOW_NAME, video_image)
        cv2.waitKey(1)



        
'''-----------------------------------------------------------------------------------------------------
This is the main function.
First, it sets up the NCS
Next, it runs the neural netowrk on all of the files stored in valid images directory
After that, it calls the looping function.
when the looping function is closed, the main function closes the NCS
'''-----------------------------------------------------------------------------------------------------
def main():
    connected_devices = mvnc.EnumerateDevices()
    if len(connected_devices) == 0:
        print("No NCS is detected")
        quit()
    
    primary_NCS = mvnc.Device(connected_devices[0])
    
    primary_NCS.OpenDevice()
    
    with open(GRAPH_FILENAME, mode = "rb") as f:
        graph_in_memory = f.read()
        
    graph = primary_NCS.AllocateGraph(graph_in_memory)
    
    valid_output = []
    for i in validated_image_list:
        valid_image = cv2.imread("./validated_images/"+i)
        valid_output.append(infer(valid_image, graph))
    run_camera(valid_output, validated_image_list, graph)
    
    graph.DeallocateGraph()
    primary_NCS.CloseDevice()
    
    
    
    
if __name__ == "__main__":
    main()