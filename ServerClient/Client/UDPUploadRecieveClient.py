# UDP Upload Recieve Client
import socket
import time

UDP_IP = "131.230.191.195"
UDP_PORT = 5005


def main():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    #sock.bind((UDP_IP, UDP_PORT))
    while(True):
        data = input('Enter your message. Enter | to quit : ')
        timerA = time.time()
        sock.sendto(data.encode(), (UDP_IP, UDP_PORT))
        timerB = time.time()
        reply, addr = sock.recvfrom(1024)
        timerC = time.time()
        print(reply.decode())
        print("Time to encode and send: " + str(timerB-timerA))
        print("Time to recieve and send: " + str(timerC-timerB))
        if(data == '|'):
            break
        print("Done")
    sock.close()
    
if __name__ == "__main__":
    main()