import numpy as np
import cv2

RESOLUTION_WIDTH = 1920
RESOLUTION_HEIGHT = 1080

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, RESOLUTION_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, RESOLUTION_HEIGHT)
 
    # Check if camera opened successfully
    if (cap.isOpened() == False): 
      print("Unable to read camera feed")
 
    # Default resolutions of the frame are obtained.The default resolutions are system dependent.
    # We convert the resolutions from float to integer.
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
 
    # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
    out = cv2.VideoWriter('outpy.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
 
    while(True):
        ret, frame = cap.read()
 
        if ret == True: 
     
    # Write the frame into the file 'output.avi'
            out.write(frame)
 
    # Display the resulting frame    
            cv2.imshow('frame',frame)
 
    # Press Q on keyboard to stop recording
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
  # Break the loop
        else:
            break 
 
# When everything done, release the video capture and video write objects
    cap.release()
    out.release()
 
# Closes all the frames
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()