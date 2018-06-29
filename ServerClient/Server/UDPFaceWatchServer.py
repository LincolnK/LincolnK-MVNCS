# UDP Face Watch Video Server

import socket
import numpy as np
import cv2
import time

UDP_IP = "131.230.191.195"
UDP_PORT = 5005
BUFFER_SIZE = 4096

DEBUG = False



def rcvImage(sock):
	data = []
	packet_length = -1
	addr = None
	
	if(DEBUG):
		print("Attempting to recieve")
	while(True):
		if(DEBUG):
			print("W1")
		len_packet, addr = sock.recvfrom(BUFFER_SIZE)
		pkt_tag = len_packet[:4].decode('utf-8', 'ignore')
		if(pkt_tag == "LEN*"):
			packet_length = int(len_packet[4:].decode('utf-8', 'ignore'))
			break
		elif(pkt_tag == "TER*"):
			return (False, None, addr)
	sock.settimeout(2)
	timeouts = 0
	while(len(data)<packet_length):
		try:
			if(DEBUG):
				print("W2")
			img_packet, addr = sock.recvfrom(BUFFER_SIZE)
			data.append(img_packet)
		except:
			if(timeouts > 10):
				print("Connection Has Died")
				return (False, None, addr)
			if(DEBUG):
				print("Lost Packet")
	sock.settimeout(None)

	
	valid = True
	image = False
	if(DEBUG):
		print("Packet Recieved")
	try:
		bstream = b"".join(data)
		image = np.loads(bstream)
		if(DEBUG):
			print("Image Verified")
	except:
		if(DEBUG):
			print("Image Failed")
		valid = False

	return (valid, image, addr) 
		
	

def reply(sock, faces, addr):
	if(DEBUG):
		print("Attempting to Reply")
	if(faces is -1):
		sock.sendto("ERR*".encode(), addr)
	elif(faces is None):
		sock.sendto("ERR*".encode(), addr)
	else:
		data = ""
		for(x,y,w,h) in faces:
			data += ("<" + str(x) + "|" + str(y) + "|" + str(w))
		sock.sendto(data.encode('utf-8'), addr)
		if(DEBUG):
			print("Reply Sent")



def main():
	sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
	sock.bind((UDP_IP, UDP_PORT))
	
	face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
	if(DEBUG):
		print("Start")
	try:
		while(True):
			timerA = time.time()
			valid, image, addr = rcvImage(sock)
			timerB = time.time()
			if(valid):
				if(DEBUG):
					print("Packet Recieved and good")
				faces = face_cascade.detectMultiScale(image, 1.3, 5)
				timerC = time.time()
				reply(sock, faces, addr)
				timerD = time.time()
				print("-----------------------------------------")
				print("RCV time: " + str(timerB - timerA))
				print("Cascade time: " + str(timerC - timerB))
				print("Reply time: " + str(timerD - timerB))
			else:

				if(image is False):
					if(DEBUG):
						print("Recieved Bad packet")
					reply(sock, -1, addr)
				else:
					print("Comunication Terminated")
					sock.close()
					break

	except KeyboardInterrupt:
		sock.close()
		print("Done")
	
	except:
		sock.close()
		print("Error")
		raise
		
	

if __name__ == "__main__":
	main()
