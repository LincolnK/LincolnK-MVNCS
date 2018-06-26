import sys
sys.path.insert(0, "ncapi2_shim")
import mvnc_simple_api as mvnc

import csv
import time
import numpy
import cv2
import os

validated_image_list = os.listdir("./validated_images/")
GRAPH_FILENAME = "facenet_celeb_ncs.graph"
CV_WINDOW_NAME = "Lincoln Kinley's Face ID app"
CSV_FILENAME = "data.csv"

CAMERA_INDEX = 0
RESOLUTION_WIDTH = 640  # was 640, tested at 720 turned blue, tested at 1280 couldn't recognize faces
RESOLUTION_HEIGHT = 480   # was 480, tested at 1280 turned blue, tested at 720 couldn't recognize faces
NETWORK_WIDTH = 160       # was 160
NETWORK_HEIGHT = 160      # was 160

MATCH_THRESHOLD = 0.4




'''-----------------------------------------------------------------------------------------------------
This function is where the neural network is actually put into play.
Preproccess function sets up the image so that it can properly be read by the neural network.
LoadTensor is the function that simulates the Neural Network

Parameters
    image (type mat)
        This is the image that was taken by the camera, sent in whatever resolution the camera took the picture at
    graph (type graph)
        This is the .graph file that the Neural Network uses to find facial features       
'''
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
'''
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
'''
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
'''
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
'''
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
'''
def run_camera(output, valid_image_name, graph, csvw, width, height, filename):
    video_filename = filename
    video = cv2.VideoCapture(video_filename)
    
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    
    if(not video.isOpened()):
        print("Failed to open the video file: " + video_filename)
        return
    
    cv2.namedWindow(CV_WINDOW_NAME)
    
    detection_frames = 0
    total_frames = 0
    assurance = 0
    
    timerClist = []
    timerDlist = []
    timerElist = []
    
    readTime = []
    cascadeTime = []
    neuralTime = []
    matchTime = []
    showTime = []
    cycleTimeNoFace = []
    cycleTimeOneFace = []
    cycleTimeTwoFace = []
    cycleTimeThreeFace = []
    
    timerA = None
    timerB = None
    timerC = None
    timerD = None
    timerE = None
    timerF = None
    timerG = None
    timerH = None
    currentTime = time.time()
    
    while True:
        # Read an image from the stream of images in the mp4 file, then resize it to the resolution being tested
        timerA = time.time()
        facesDetected = 0
        return_val, video_image = video.read()
        if (not return_val):
            print("No Image from camera")
            break
        video_image = cv2.resize(video_image, (width, height))
        
        # Use Haar Cascade to find the bounding boxes of faces in the image
        timerB = time.time()
        gray = cv2.cvtColor(video_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        timerH = time.time()
        set = True
        for(x,y,w,h) in faces:
            facesDetected = len(faces)
            if(set):
                set = False
                detection_frames += 1
            # Test each face in the nerual network
            timerC= time.time()
            cropped_face = video_image[y:y+h, x:x+w]
            test_output = infer(cropped_face, graph)
            
            # Compaire the detected face with the faces in the validated folder
            timerD = time.time()
            min_distance = 100
            min_index = -1
            for i in range(0,len(output)):
                distance = face_match(output[i], test_output)
                if distance < min_distance:
                    min_distance = distance
                    min_index = i
            if(min_distance<=MATCH_THRESHOLD):
                frame_name = validated_image_list[min_index]
                print("Found! " + validated_image_list[min_index])
                cv2.rectangle(video_image,(x,y),(x+w,y+h),(255,0,0),2)
                cv2.putText(video_image, validated_image_list[min_index], (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            else:
                frame_name = "No Verified Face"
                cv2.rectangle(video_image,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(video_image, "Unknown", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                print("Fail!")
            if(min_distance < 1.4):
                assurance = assurance + min_distance
            timerE = time.time()
            neuralTime.append(timerD - timerC)
            matchTime.append(timerE - timerD)
            
        timerF = time.time()
        prop_val = cv2.getWindowProperty(CV_WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO)
        if(prop_val < 0.0):
            print("Closed")
            break
        # Show the Image
        total_frames += 1
        cv2.imshow(CV_WINDOW_NAME, video_image)
        cv2.waitKey(1)
        timerG = time.time()
        if(facesDetected == 0):
            cycleTimeNoFace.append(timerG-timerA)
        elif(facesDetected == 1):
            cycleTimeOneFace.append(timerG-timerA)
        elif(facesDetected == 2):
            cycleTimeTwoFace.append(timerG-timerA)
        elif(facesDetected == 3):
            cycleTimeThreeFace.append(timerG-timerA)
        readTime.append(timerB - timerA)
        cascadeTime.append(timerH - timerB)
        showTime.append(timerG - timerF)
        
    try:
        avg_readTime = str(sum(readTime)/float(len(readTime)))
    except:
        avg_readTime = "NA"
        
    try:
        avg_cascadeTime = str(sum(cascadeTime)/float(len(cascadeTime)))
    except:
        avg_cascadeTime = "NA"
        
    try:
        avg_neuralTime = str(sum(neuralTime)/float(len(neuralTime)))
    except:
        avg_neuralTime = "NA"
        
    try:
        avg_matchTime = str(sum(matchTime)/float(len(matchTime)))
    except:
        avg_matchTime = "NA"
        
    try:
        avg_showTime = str(sum(showTime)/float(len(showTime)))
    except:
        avg_showTime = "NA"
        
    avg_cycleTimeNoFace = 'NA'
    avg_cycleTimeOneFace = 'NA'
    avg_cycleTimeTwoFace = 'NA'
    avg_cycleTimeThreeFace = 'NA'
    if(len(cycleTimeNoFace) != 0):
        avg_cycleTimeNoFace = str(sum(cycleTimeNoFace)/float(len(cycleTimeNoFace)))
    if(len(cycleTimeOneFace) != 0):
        avg_cycleTimeOneFace = str(sum(cycleTimeOneFace)/float(len(cycleTimeOneFace)))
    if(len(cycleTimeTwoFace) != 0):
        avg_cycleTimeTwoFace = str(sum(cycleTimeTwoFace)/float(len(cycleTimeTwoFace)))
    if(len(cycleTimeThreeFace) != 0):
        avg_cycleTimeThreeFace = str(sum(cycleTimeThreeFace)/float(len(cycleTimeThreeFace)))
    
    average_assurance = "NA"
    if(detection_frames != 0):
        average_assurance = str(assurance/detection_frames)
    
    csvw.writerow([str(width) + "x" + str(height) + ";" + filename + ";" + str(total_frames) + ";" + str(detection_frames) + ";" + str(assurance) + ";" + average_assurance + ";" + avg_readTime + ";" + avg_cascadeTime + ";" + avg_neuralTime + ";" + avg_matchTime + ";" + avg_showTime + ';' + avg_cycleTimeNoFace + ';' + avg_cycleTimeOneFace + ';' + avg_cycleTimeTwoFace + ';' + avg_cycleTimeThreeFace])
    video.release()



        
'''-----------------------------------------------------------------------------------------------------
This is the main function.
First, it sets up the NCS
Next, it runs the neural netowrk on all of the files stored in valid images directory
After that, it calls the looping function.
when the looping function is closed, the main function closes the NCS
'''
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
    
    with open('timing_and_accuracy_analysis.csv', 'w', newline='') as csvfile:
        csvw = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        csvw.writerow(["Resolution;Test;Total Frames;Detected Frames;Assurance;Average Assurance;Average Read Time;Average Cascade Time;Average Neural Time;Average Match Time;Average Show Time;Average Cycle Time No Face;Average Cycle Time One Face;Average Cycle Time Two Face;Average Cycle Time Three Face"])
        run_camera(valid_output, validated_image_list, graph, csvw, 640, 480, "3ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1280, 720, "3ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1920, 1080, "3ft_test.mp4")
        
        run_camera(valid_output, validated_image_list, graph, csvw, 640, 480, "5ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1280, 720, "5ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1920, 1080, "5ft_test.mp4")
        
        run_camera(valid_output, validated_image_list, graph, csvw, 640, 480, "10ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1280, 720, "10ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1920, 1080, "10ft_test.mp4")
        
        run_camera(valid_output, validated_image_list, graph, csvw, 640, 480, "15ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1280, 720, "15ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1920, 1080, "15ft_test.mp4")
        
        run_camera(valid_output, validated_image_list, graph, csvw, 640, 480, "20ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1280, 720, "20ft_test.mp4")
        run_camera(valid_output, validated_image_list, graph, csvw, 1920, 1080, "20ft_test.mp4")

    
    cv2.destroyAllWindows()
    graph.DeallocateGraph()
    primary_NCS.CloseDevice()
    
    
    
    
if __name__ == "__main__":
    main()