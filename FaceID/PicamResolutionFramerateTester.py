import cv2
import time

TEST_TIME = 60

def runTest(camera, testName):
    actual_camera_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_camera_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frames = 0
    prevTime = None
    startTime = time.time()
    currentTime = startTime
    while True:
        det, image = camera.read()
        prevTime = currentTime
        currentTime = time.time()
        frames +=1
        framerate = str(round(1/(currentTime-prevTime),1))
        cv2.putText(image, framerate, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.imshow(testName, image)
        cv2.waitKey(1)
        if(currentTime-startTime >= TEST_TIME):
            break
    avgFramerate = str(frames/(currentTime - startTime))
    print(testName + "\n  - Width: " + str(actual_camera_width) + "\n  - Height: " + str(actual_camera_height) + " \n  - Average Framerate: " + avgFramerate)
    file = open("ResolutionBenchmarks.txt", "a")
    file.write(testName + "\n  - Width: " + str(actual_camera_width) + "\n  - Height: " + str(actual_camera_height) + " \n  - Average Framerate: " + avgFramerate + "\n")
    file.close()
    

def main():
    camModel = input("Please enter the camera model being used: ")
    file = open("ResolutionBenchmarks.txt", "a")
    file.write(camModel + "\n\n")
    file.close()
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
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "480p Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p480_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p480_H)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "720p Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p720_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p720_H)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "1080p Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p1080_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p1080_H)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    
    testName = "360p Rotated Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p360_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p360_W)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "480p Rotated Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p480_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p480_W)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "720p Rotated Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p720_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p720_W)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "1080p Rotated Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, p1080_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, p1080_W)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Standard Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_W)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_H)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID High Res Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_W*2)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_H*2)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Low Res Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_W/2)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_H/2)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Rotated Standard Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_H)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_W)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Rotated High Res Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_H*2)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_W*2)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "FaceID Rotated Low Res Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, faceID_H/2)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, faceID_W/2)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    
    testName = "Network Test"
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, network_Res)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, network_Res)
    cv2.namedWindow(testName)
    runTest(camera, testName)
    cv2.destroyAllWindows()
    file = open("ResolutionBenchmarks.txt", "a")
    
    notes = input("Enter any notes: ")
    file.write("\n"+ notes)
    file.write("\n\n ------------------------------------------------------- \n")
    file.close()
    
    camera.release()


if __name__ == "__main__":
    main()