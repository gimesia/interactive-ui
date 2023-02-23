import random
import time
import asyncio
import cv2 as cv
import numpy as np
from zmq import Context, Socket
import zmq

PING = 'PING'
THRESHOLD_INFO = 'THRESHOLD_INFO'
INP_IMAGE = 'INP_IMAGE'
QUERY_CONTOUR = 'QUERY_CONTOUR'
REQ_RECEIVED = 'REQUEST_RECEIVED'
DONE = 'DONE'
FAILED = 'FAILED'
ALIVE = 'ALIVE'
EXIT = 'EXIT'
UNKNOWN = 'UNKNOWN'


class Cluster():
    """ Class for a class of identified objects
    """

    def __init__(self, clustername: str, color):
        self.name = clustername
        self.color: tuple(int, int, int) = color
        self.checked = True
        self.contours = []
        self.disabled_contours = []

    def get_all_contours(self):
        return np.concat(self.contours, self.disabled_contours)

    def disable_contour(self, index: int):
        self.disabled_contours.append(self.contours.pop(index))

    def enable_contour(self, index: int):
        self.contours.append(self.disabled_contours.pop(index))

    @property
    def __geo_interface__(self):
        features = []
        for i, contour in enumerate(self.contours):
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": contour[:, 0, :]},
                "properties": {
                    "index": i,
                    "area": cv.contourArea(contour),
                    "center": 0  # centroid_for_contour(contour)
                }})
        return {"type": "FeatureCollection", "features": features, "properties": {"cluster": self.name, "index": i}}


class ImageWindow():
    def __init__(self, name="Interactive UI"):
        self.name = name
        self.clusters: list[Cluster] = [Cluster("Cl1", (57, 106, 177)), Cluster("Cl2", (107, 76, 154)), Cluster(
            "Cl3", (204, 37, 41)), Cluster("Cl4", (62, 150, 81))]
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.th_c = 121
        self.blocksize = 121
        self.show_contours = True
        self.refresh_on_next = False
        self.disable_all = False
        self.edit = True

        cv.namedWindow(self.name)
        # Mouse event callbacks
        cv.setMouseCallback(self.name, self.on_mouse_event)
        self.create_ui_controls()

    def set_base_image(self, img: np.ndarray) -> None:
        self.og_img = img.copy()
        self.contour_img = img.copy()

    def show_img(self) -> None:
        if self.show_contours:
            cv.imshow(self.name, self.contour_img)
        else:
            cv.imshow(self.name, self.og_img)

    def ath(self, block: int = 151, c: int = 150):
        img = self.og_img.copy()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        print(f"c: {self.th_c}, bc: {self.blocksize}\nATH")

        th = cv.adaptiveThreshold(
            gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, self.blocksize, self.th_c)

        contours, h = cv.findContours(
            th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contours_map = list(map(lambda x: cv.contourArea(x), contours))

        if not len(contours_map):
            return

        mx, mn = max(contours_map), min(contours_map)
        diff = (mx - mn) / 4

        for cl in self.clusters:
            cl.contours = []
            cl.disabled_contours = []

        for i, cont in enumerate(contours):
            if contours_map[i] < mn + diff:
                self.clusters[0].contours.append(cont)
            elif contours_map[i] < mn + 2 * diff:
                self.clusters[1].contours.append(cont)
            elif contours_map[i] < mn + 3 * diff:
                self.clusters[2].contours.append(cont)
            else:
                self.clusters[3].contours.append(cont)
        print("ATH finished")
        return c

    def on_toggle_cluster(*args):
        print("on_toggle_cluster:")
        print(args)
        pass

    def on_toggle_mode(*args):
        print("on_toggle_mode:")
        print(args)
        pass

    def on_disable_all(*args):
        print("on_disable_all:")
        print(args)
        pass

    def on_display_stats(*args):
        print("on_display_stats:")
        print(args)
        pass

    def save_data(*args):
        print("save_data:")
        print(args)
        pass

    def set_block_size(self, val, *args):
        print(f"set_block_size: {val}")
        if val % 2 == 0:
            val += 1
        self.blocksize = val
        pass

    def set_c_value(self, val, *args):
        print(f"set_c_value: {val}")
        self.th_c = val
        pass

    def create_ui_controls(self):
        cv.createButton("EDIT MODE", self.on_toggle_mode, "EDIT",
                        cv.QT_RADIOBOX, self.edit)
        cv.createButton("INFO MODE", self.on_toggle_mode,
                        "INFO", cv.QT_RADIOBOX, not self.edit)

        for cl in self.clusters:
            cv.createButton(cl.name, self.on_toggle_cluster, cl.name,
                            cv.QT_CHECKBOX | cv.QT_NEW_BUTTONBAR, cl.checked)

        cv.createButton("EN/DISABLE ALL", self.on_disable_all, "ALL",
                        cv.QT_PUSH_BUTTON | cv.QT_NEW_BUTTONBAR, 0)

        cv.createButton("DISPLAY STATS", self.on_display_stats, "DISPLAY",
                        cv.QT_PUSH_BUTTON | cv.QT_NEW_BUTTONBAR, 0)

        cv.createButton("SAVE DATA", self.save_data, "print",
                        cv.QT_PUSH_BUTTON)

        cv.createTrackbar('Block size:', self.name,
                          self.blocksize, 255, self.set_block_size)

        cv.createTrackbar('C value:', self.name,
                          self.th_c, 255, self.set_c_value)

    def update_contour_img(self) -> None:
        """

        """
        im = self.og_img.copy()

        self.th_c = random.choice(range(10, 121))

        self.blocksize = random.choice(range(100, 191))
        if self.blocksize % 2 == 0:
            self.blocksize -= 1

        for cl in self.clusters:
            cv.drawContours(im, cl.contours, -1, cl.color, 1)
            cv.drawContours(im, cl.disabled_contours, -1, (100, 100, 100), 1)
        # self.contour_img = im
        print("UPDATE contour img")
        return im

    def on_mouse_event(self, event, x: int, y: int, *args) -> None:
        """Mouse event listener with all the respective actions to be listened to (click, dblclick, hover, etc.)

        Args:
            event (str): type of the mouse event
            x (int): x coord
            y (int): y coord
        """
        if self.og_img is None or self.contour_img is None or not event:
            return

        point = (x, y)
        print(event)

        if event == 4:
            self.ath()
            self.contour_img = self.update_contour_img()
            self.refresh_on_next = True


class BigTing():
    def __init__(self):
        self.context: Context = Context()
        self.sock: Socket = self.context.socket(zmq.PAIR)
        self.sock.bind("tcp://*:5557")
        self.op = True
        self.window = ImageWindow()

    def handle_message(self, msg: str):
        print(f"Command: {msg}, redirecting accordingly")
        if msg == PING:
            self.pong()
        elif msg == INP_IMAGE:
            self.receive_image()
        elif msg == EXIT:
            self.confirm_exit(True)
        else:
            self.sock.send_string(f"{msg}")

    def pong(self):  # Answer to ping
        self.sock.send_string(ALIVE)
        print(PING)

    def confirm_req(self):  # Sends confirmation of received request
        print("Sending confirmation of received request")
        self.sock.send_string(REQ_RECEIVED)

    def confirm_req_complete(self):  # Sends confirmation of received request
        print("Sending confirmation")
        self.sock.send_string(DONE)

    def confirm_req_failed(self):  # Sends receit of failed request
        print("Sending fail receipt")
        self.sock.send_string(FAILED)

    def confirm_exit(self, close=True):  # Confirms EXIT command
        print("Sending EXIT confirmation")
        if close:
            self.op = False
        self.sock.send_string("q")

    def receive_image(self):  # Reception of image from socket
        print("Receiving image")
        self.confirm_req()
        message = self.sock.recv_pyobj()
        try:
            self.window.set_base_image(message)
            self.confirm_req_complete()
        except:
            self.confirm_req_failed()
        self.window.ath()
        self.window.update_contour_img()
        self.window.refresh_on_next = True

    """ZeroMQ communication async coroutine"""
    async def coroutine_zmq(self):
        print('ZMQ Coroutine is running')
        i = 0
        while self.op:
            i += 1
            message = None
            print(f'zmq iter: {i}')
            if self.sock.poll(100, zmq.POLLIN):
                message = self.sock.recv_string()
                self.handle_message(message)
                print(f'message: \"{message}\"')
            await asyncio.sleep(0)

        self.sock.close()
        await asyncio.sleep(0)
        print('ZMQ Coroutine is done')

    """OpenCV image display coroutine"""
    async def coroutine_image(self):
        print('Image Coroutine')

        j = 0
        while self.op:
            j += 1
            print(f'image iter: {j}')

            # Showing Image
            if self.window.og_img is not None and self.window.disable_all:
                cv.imshow(self.window.name, self.window.og_img)

            if self.window.refresh_on_next:
                cv.imshow(self.window.name, self.window.contour_img)
                self.window.refresh_on_next = False

            key = cv.waitKey(500)

            # Breaks infinite loop if SPACE is pressed OR OpenCV window is closed
            if key == 32 or cv.getWindowProperty(self.window.name, cv.WND_PROP_VISIBLE) < 1:
                self.op = False
                break

            await asyncio.sleep(0)
        cv.destroyAllWindows()

        # Awaiting end
        await asyncio.sleep(0)
        print('Image Coroutine is done')

    async def main(self):
        # create the coroutines
        coroutine1 = self.coroutine_zmq()
        coroutine2 = self.coroutine_image()

        # schedule the coroutine to run in the background
        task1 = asyncio.create_task(coroutine1)
        task2 = asyncio.create_task(coroutine2)

        # simulate continue on with other things
        await task1
        await task2


if __name__ == "__main__":
    # run the asyncio program
    a = BigTing()

    asyncio.run(a.main())
