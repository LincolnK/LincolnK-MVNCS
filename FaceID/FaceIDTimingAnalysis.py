#Face ID With Timing Analysis

import sys
sys.path.insert(0, "ncapi2_shim")
import mvnc_simple_api as mvnc

import numpy
import cv2
import os
import time

validated_image_list = os.listdir("./validated_images/")
GRAPH_FILENAME = "facenet_celeb_ncs.graph"
CV_WINDOW_NAME = "Lincoln Kinley's Face ID app"

CAMERA_INDEX = 0
RESOLUTION_WIDTH = 512*2   # was 640, tested at 720 turned blue, tested at 1280 couldn't recognize faces
RESOLUTION_HEIGHT = 640*2   # was 480, tested at 1280 turned blue, tested at 720 couldn't recognize faces
NETWORK_WIDTH = 160       # was 160
NETWORK_HEIGHT = 160      # was 160

MATCH_THRESHOLD = 0.8

TEST_TIME = 60


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
    return total_difference


def handle_keys(raw_key):
    ascii_code = raw_key & 0xFF
    if ((ascii_code == ord('q')) or (ascii_code == ord('Q'))):
        return False

    return True


def runTest(output, valid_image_name, graph, camera, testName):

    if((camera == None) or (not camera.isOpened())):
        print("No Camera has been detected!")
        return
    
    actual_camera_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_camera_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    frames = 0
    framerate = 0
    
    readTime = []
    neuralTime = []
    matchTime = []
    showTime = []
    totalTime = []
    totalFramerate = []
    
    timerA = None
    timerB = None
    timerC = None
    timerD = None
    timerE = None
    
    match=False
    
    prevTime = None
    startTime = time.time()
    while True:
        timerA = time.time()
        
        return_val, video_image = camera.read()
        if (not return_val):
            print("No Image from camera")
            break
        frames += 1
        
        timerB = time.time()
        test_output = infer(video_image, graph)
        
        frame_name = str(framerate)
        
        timerC = time.time()
        min_distance = 100
        min_index = -1
        
        for i in range(0,len(output)):
            distance = face_match(output[i], test_output)
            if distance < min_distance:
                min_distance = distance
                min_index = i
        if(min_distance<=MATCH_THRESHOLD):
            found_match = True
        else:
            found_match = False
        timerD = time.time()
        frame_update(video_image, frame_name, found_match)

        cv2.imshow(testName, video_image)
        cv2.waitKey(1)
        timerE = time.time()
        framerate = round(1/(timerE-timerA),1)
        
        readTime.append(timerB-timerA)
        neuralTime.append(timerC-timerB)
        matchTime.append(timerD-timerC)
        showTime.append(timerE-timerD)
        totalTime.append(timerE-timerA)
        totalFramerate.append(framerate)
        
        if(timerE-startTime >= TEST_TIME):
            break
        
    avg_readTime = str(sum(readTime)/float(len(readTime)))
    avg_neuralTime = str(sum(neuralTime)/float(len(neuralTime)))
    avg_matchTime = str(sum(matchTime)/float(len(matchTime)))
    avg_showTime = str(sum(showTime)/float(len(showTime)))
    avg_totalTime = str(sum(totalTime)/float(len(totalTime)))
    avg_framerate = str(sum(totalFramerate)/float(len(totalFramerate)))

    file = open("FaceID_Test_Results-2F.txt", "a")
    file.write(testName + "\n  - Width: " + str(actual_camera_width) + "\n  - Height: " + str(actual_camera_height) + " \n  - Average Framerate: " + avg_framerate + "\n  - Frames: " + str(frames))
    file.write("\n  - Total Time: " + avg_totalTime + "\n  - Read Time: " + avg_readTime + "\n  - Neural Time: " + avg_neuralTime + "\n  - Match Time: " + avg_matchTime)
    file.write("\n  - Show Time: " + avg_showTime + "\n\n")
    file.close()
        
        
def main():
    camModel = input("Please enter the camera model being used: ")
    file = open("FaceID_Test_Results-2F.txt", "a")
    file.write(camModel + "\n\n")
    file.close()
    
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
        
        
    p360_H = 360
    p360_W = 480
    
    p480_H = 480
    p480_W = 640
    
    p720_H = 720
    p720_W = 1280
    
    p1080_H = 1080
    p1080_W = 1920
    
    faceID_H = 640
    faceID_W = 512
    
    network_Res = 160
    
    camera = cv2.VideoCapture(1)
    
    testName = "360p Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p360_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p360_H)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "480p Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p480_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p480_H)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "720p Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p720_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p720_H)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "1080p Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p1080_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p1080_H)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    
    testName = "360p Rotated Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p360_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p360_W)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "480p Rotated Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p480_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p480_W)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "720p Rotated Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p720_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p720_W)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "1080p Rotated Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p1080_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p1080_W)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Standard Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_H)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID High Res Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_W*2)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_H*2)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Low Res Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_W/2)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_H/2)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Rotated Standard Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_W)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Rotated High Res Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_H*2)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_W*2)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Rotated Low Res Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_H/2)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_W/2)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    
    testName = "Network Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, network_Res)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, network_Res)
    cv2.namedWindow(testName)
    runTest(valid_output, validated_image_list, graph, camera, testName)
    cv2.destroyAllWindows()
    file = open("FaceID_Test_Results-2F.txt", "a")
    
    notes = input("Enter any notes: ")
    file.write("\n"+ notes)
    file.write("\n\n ------------------------------------------------------- \n")
    file.close()
    
    camera.release()
    
    graph.DeallocateGraph()
    primary_NCS.CloseDevice()
    
if __name__ == "__main__":
    main()