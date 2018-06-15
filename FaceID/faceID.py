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

MATCH_THRESHOLD = 0.15


def infer(image, graph):
    resized_image = preprocess(image)
    graph.LoadTensor(resized_image.astype(numpy.float16), None)
    output, userobj = graph.GetResult()
    return output
    
    
def frame_update(image, info, matching):
    rect_width = 10
    offset = int(rect_width/2)
    if(info != None):
        cv2.putText(image, info, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    if(matching):
        cv2.rectangle(image, (offset, offset), (image.shape[1]-offset-1, image.shape[0]-offset-1), (0, 255, 0), 10)
    else:
        cv2.rectangle(image, (offset, offset), (image.shape[1]-offset-1, image.shape[0]-offset-1), (0, 0, 255), 10)


def whiten_image(image):
    mean = numpy.mean(image)
    std = numpy.std(image)
    std_adjusted = numpy.maximum(std, 1.0 / numpy.sqrt(image.size))
    whitened_image = numpy.multiply(numpy.subtract(image, mean), 1 / std_adjusted)
    return whitened_image


def preprocess(image):
    preprocessed_image = cv2.resize(image, (NETWORK_WIDTH, NETWORK_HEIGHT))
    preprocessed_image = cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB)
    preprocessed_image = whiten_image(preprocessed_image)
    return preprocessed_image


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


def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True


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
        frame_count += 1
        frame_name = "Camera Frame " + str(frame_count)
        test_output = infer(video_image, graph)
            
        min_distance = 100
        min_index = -1
            
        for i in range(0,len(output)):
            distance = face_match(output[i], test_output)
            if distance < min_distance:
                min_distance = distance
                min_index = i
        if(min_distance<=MATCH_THRESHOLD):
            if(rechecks > 10):
                print("Found! File " + frame_name + " matches " + validated_image_list[min_index])
                found_match = True
                if(rechecks > 12):
                    rechecks = 12
            else:
                print("Checking... File " + frame_name + " matches " + validated_image_list[min_index])
                found_match = False
            rechecks += 1
        else:
            found_match = False
            print("Fail! File " + frame_name + " does not match any image.")
            if(rechecks <= 10):
                rechecks = 0
            else:
                rechecks = rechecks - 1
        frame_update(video_image, frame_name, found_match)

        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if(prop_val < 0.0):
            print("Closed")
            break
        cv2.imshow(CV_WINDOW_NAME, video_image)
        raw_key = cv2.waitKey(1)
        if (raw_key != -1):
            if (handle_keys(raw_key) == False):
                print('user pressed Q')
                break
        
    if (found_match):
        cv2.imshow(CV_WINDOW_NAME, video_image)
        cv2.waitKey(0)
            
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