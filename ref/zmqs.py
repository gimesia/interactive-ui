import time
import zmq
from zmq import Socket, Context
import asyncio
import cv2 as cv


PING = 'PING'
THRESHOLD_INFO = 'THRESHOLD_INFO'
INP_IMAGE = 'INP_IMAGE'
QUERY_CONTOUR = 'QUERY_CONTOUR'
DONE = 'DONE'
ALIVE = 'ALIVE'
EXIT = 'EXIT'
UNKNOWN = 'UNKNOWN'


class ScriptQuantOut():
    def __init__(self):
        self.context: Context = Context()
        self.sock: Socket = self.context.socket(zmq.PAIR)
        self.sock.bind("tcp://*:5557")
        self.win = None
        self.is_open = True
        self.img = cv.imread("cells_6.jpg")

    def pong(self):
        self.sock.send_string(ALIVE)
        print(PING)

    def confirm_req(self):
        self.sock.send_string(DONE)

    def confirm_exit(self, close=True):
        self.is_open = False
        if close:
            cv.destroyAllWindows()
        self.sock.send_string("q")

    def receive_image(self):
        self.confirm_req()
        message = self.sock.recv_pyobj()
        print(message)

    def run_server(self):
        print("Server running")
        it = 0
        while self.is_open:
            print("Server running")

            self.open_img(it)
            key = cv.waitKey(1000)
            print(key)

            message = None

            if self.sock.poll(100, zmq.POLLIN):
                message = self.sock.recv_string()
                if message == PING:
                    self.pong()
                elif message == INP_IMAGE:
                    self.receive_image()
                elif message == EXIT:
                    self.confirm_exit()
                else:
                    self.sock.send_string(f"{message}")
            it += 1
            if cv.getWindowProperty(self.win, cv.WND_PROP_VISIBLE) < 1:
                print(cv.getWindowProperty(self.win, cv.WND_PROP_VISIBLE))
                print("Server stopped")
                self.confirm_exit(close=False)
        self.sock.close()

    def open_img(self, it):
        if self.win is None:
            self.win = 'A'
            cv.namedWindow(self.win)
        im = cv.putText(self.img.copy(), it.__str__(), (50, 50),
                        cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv.LINE_AA)
        cv.imshow(self.win, im)


sq = ScriptQuantOut()
sq.run_server()
