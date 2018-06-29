#UDP Upload Recieve Server

import socket

UDP_IP = "131.230.191.195"
UDP_PORT = 5005
BUFFER_SIZE= 1024

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

sock.bind((UDP_IP, UDP_PORT))

while(True):
	data, addr = sock.recvfrom(BUFFER_SIZE)
	print("recieved message: " + data.decode())
	sock.sendto(data, addr)
	if '|' in data.decode():
		sock.sendto("Bye!".encode(), addr)
		print("Delimiter Found")
		break
