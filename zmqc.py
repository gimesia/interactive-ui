import cv2 as cv
import numpy as np
import zmq

PING = 'PING'
THRESHOLD_INFO = 'THRESHOLD_INFO'
INP_IMAGE = 'INP_IMAGE'
QUERY_CONTOUR = 'QUERY_CONTOUR'
REQ_RECEIVED = 'REQUEST_RECEIVED'
DONE = 'DONE'
ALIVE = 'ALIVE'
EXIT = 'EXIT'
UNKNOWN = 'UNKNOWN'

context = zmq.Context()

#  Socket to talk to server
print("Connecting to hello world server...")
socket: zmq.Socket = context.socket(zmq.PAIR)
socket.connect("tcp://localhost:5557")


im = [cv.imread("./cells_6.jpg"), cv.imread("./img/cells4.tif"),
      cv.imread("./img/cells3.tif")]

i = 0


def send_img(sock: zmq.Socket, img: np.ndarray):
    sock.send_string("INP_IMAGE")
    message = sock.recv_string()
    print(message)
    if message == REQ_RECEIVED:
        print("Image sending intention acknowledged -> sending image")
        sock.send_pyobj(img)
        print("Waiting for response")
        message = sock.recv_string()
        if message == DONE:
            print("Image sent successfully")
        else:
            print("!Error in communicating image!")


if __name__ == "__main__":
    while True:
        inp = input()

        if inp == "im":
            send_img(socket, im[i])
            i = (i + 1) % len(im)
            continue
        elif inp == "q":
            socket.send_string("EXIT")
        print(f"Sending request...")
        socket.send_string(inp)
        message = socket.recv_string()
        print(f"Received reply: \"{message}\"")
        if inp == "q" or message == "q":
            break
