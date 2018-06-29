import socket
import numpy as np
import cv2

UDP_IP = "131.230.191.195"
UDP_PORT = 5005
BUFFER_SIZE = 1024



sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
data = []
i = 0
run = True
try:
	while(run):
		len_packet, addr = sock.recvfrom(BUFFER_SIZE)
		len_data = len_packet.decode()
		print(len_data)
		if('|' == len_data[-1:]):
			packet_length = int(len_data[0:-1])-1
			print(packet_length)

			while(True):
				img_packet, addr = sock.recvfrom(BUFFER_SIZE)
		

				data.append(img_packet)
				i+=1
				if(len(data)>packet_length):
					print("Length Reached")
					run = False
					break


	bstream = b"".join(data)
	print(i)
	image = np.loads(bstream)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for(x,y,w,h) in faces:
		cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)

	cv2.imshow("Image",image)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
except:
	print("Error")
	print("Last Packet: " + str(i))


sock.close()


