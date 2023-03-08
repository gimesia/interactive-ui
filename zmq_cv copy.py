import asyncio
import random
import threading
import time
import tkinter as tk
import cv2 as cv
import numpy as np
import pandas as pd
from zmq import Context, Socket
import zmq
from PIL import Image, ImageTk

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

ZMQ_SERVERNAME = "tcp://*:5557"


def start_new_thread(fc):
    thread = threading.Thread(target=fc)
    thread.start()


class ObjectParams():
    def __init__(self, cluster_name, i, area, center, disabled):
        self.cluster = cluster_name
        self.indx = i
        self.area = area
        self.center = center
        self.disabled = disabled

    def __str__(self) -> str:
        return f"cluster: {self.cluster}{' (disabled)' if self.disabled else ''}; index: {self.indx}; area: {self.area}; center: {self.center}"

    def put_params_on_image(self, img: np.ndarray, org: "tuple(int, int)", *kwargs) -> np.ndarray:
        lines = self.__str__().split(sep="; ")
        lines.pop(1)
        image, textbox = put_textbox_on_img(
            img, lines, (org[0] + 10, org[1]), 175)
        return image


class Cluster():
    """ Class for a class of identified objects
    """

    def __init__(self, clustername: str, color):
        self.name = clustername
        self.color: tuple(int, int, int) = color
        self.checked = True
        self.contours = []
        self.disabled_contours = []

    def contour_count(self) -> int:
        return len(self.contours) + len(self.disabled_contours)

    def disable_contour(self, index: int) -> None:
        self.disabled_contours.append(self.contours.pop(index))

    def enable_contour(self, index: int) -> None:
        print("enable contour")
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
        self.clusters: list[Cluster] = [
            Cluster("Cl1", (250, 106, 17)),
            Cluster("Cl2", (107, 76, 254)),
            Cluster("Cl3", (204, 137, 241)),
            Cluster("Cl4", (62, 150, 81))
        ]
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.th_c = 121
        self.blocksize = 121
        self.show_contours = True
        self.refresh_on_next = False
        self.edit = True
        self.stats = None

        cv.namedWindow(self.name)

        self.set_base_image(cv.imread("cells_6.jpg"))
        self.show_img()

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

    def toggle_contours(self):
        self.show_contours = not self.show_contours
        self.refresh_on_next = True

    def ath(self, block: int = 151, c: int = 150):
        img = self.og_img.copy()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

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
        return c

    def on_toggle_cluster(self, *args):
        print("on_toggle_cluster:")
        print(args)
        pass

    def on_toggle_mode(self, *args):
        print("on_toggle_mode:")
        self.edit = not self.edit
        self.refresh_on_next = True

    def on_display_stats(self, *args):
        # TODO!
        print("on_display_stats:")
        print(args)

    def save_data(self, *args):
        # TODO!
        print("save_data:")
        print(args)
        pass

    def set_block_size(self, val, *args):
        print(f"set_block_size: {val}")
        if val % 2 == 0:
            if val > 1:
                val += 1
            else:
                val = 3
        self.blocksize = val
        self.update_contour_img()

    def set_c_value(self, val, *args):
        print(f"set_c_value: {val}")
        self.th_c = val
        self.update_contour_img()

    def create_ui_controls(self):
        cv.createTrackbar('Block size:', self.name,
                          self.blocksize, 255, self.set_block_size)

        cv.createTrackbar('C value:', self.name,
                          self.th_c, 255, self.set_c_value)

    def update_contour_img(self, segment=True) -> None:
        im = self.og_img.copy()
        if segment:
            self.ath()

        for cl in self.clusters:
            if not cl.checked:
                continue
            cv.drawContours(im, cl.contours, -1, cl.color, 1)
            cv.drawContours(im, cl.disabled_contours, -1, (100, 100, 100), 1)
        self.contour_img = im

        self.refresh_on_next = True
        return im

    def extract_stats(self, *args):  # -> pd.DataFrame:
        """Extracts the summary of the clusters' distributions
        """
        lengths = np.asarray(
            list(map(lambda cl: len(cl.contours), self.clusters))
        )

        enabled = np.sum(lengths)

        disabled = np.sum(
            list(map(lambda cl: len(cl.disabled_contours), self.clusters))
        )

        all = np.sum(
            list(map(lambda cl: cl.contour_count(), self.clusters))
        )

        indexes = list(map(lambda cl: cl.name, self.clusters))

        if not all:
            frame = {'Count': np.zeros(
                len(lengths),), 'Percentage': np.zeros(len(lengths),)}
        else:
            frame = {'Count': pd.Series(lengths, index=indexes),
                     'Percentage': pd.Series((lengths / enabled) * 100, index=indexes)}

        df1 = pd.DataFrame(frame, index=indexes)
        df2 = pd.DataFrame({'Count': [enabled, disabled, all],
                            'Percentage': [100*enabled/all, 100*disabled/all, int(100), ]}, index=['enabled', 'disabled', 'all', ])
        res = pd.concat(objs=[df1, df2])

        self.stats = res

        lines = []
        for index, row in res.iterrows():
            lines.append(
                f"{index}: {int(row['Count'])} ({round(row['Percentage'], 2) if bool(row['Count']) else '0'}%)")

        return res, lines

    def claster_by_name(self, cluster_name: str) -> Cluster:
        index = list(map(lambda x: x.name, self.clusters)).index(cluster_name)
        print(self.clusters[index])
        return self.clusters[index]

    def change_cluster(self, params: ObjectParams, backwards=False):
        if params.disabled:
            return
        cl = self.claster_by_name(params.cluster)
        next = self.clusters.index(cl)
        if backwards:
            next -= 1
        else:
            next += 1
        next %= len(self.clusters)

        self.clusters[next].contours.append(
            cl.contours.pop(params.indx)
        )

    def on_mouse_event(self, event, x: int, y: int, flags, *args) -> None:
        """Mouse event listener with all the respective actions to be listened to (click, dblclick, hover, etc.)


        Args:
            event (str): type of the mouse event
            x (int): x coord
            y (int): y coord
        """
        if self.og_img is None or self.contour_img is None or not event:
            return

        point = (x, y)
        print(f"event: {event}")
        print(f"flags: {flags}")

        if event == 4:
            if flags == 17:
                # shift
                a = find_object_for_point(point, self.clusters, False)
                self.change_cluster(a, backwards=True)
            elif flags == 9:
                # ctrl
                a = find_object_for_point(point, self.clusters, False)
                self.change_cluster(a, backwards=False)
            else:
                a = find_object_for_point(point, self.clusters, True)

            self.refresh_on_next = True


class BigTing():
    def __init__(self):
        self.context: Context = Context()
        self.sock: Socket = self.context.socket(zmq.PAIR)
        self.sock.bind(ZMQ_SERVERNAME)
        self.op = True
        self.stats_img = None
        self.window = ImageWindow()
        self.tk = None
        self.tk_photo = None

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

    async def coroutine_zmq(self):
        """ZeroMQ communication async coroutine"""
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

    async def coroutine_image(self):
        """OpenCV image display coroutine"""
        print('Image Coroutine')
        start_new_thread(self.coroutine_controls)
        j = 0
        while self.op:
            j += 1
            print(f'image iter: {j}')

            # Showing Image
            if self.window.og_img is not None:
                if self.window.show_contours:
                    cv.imshow(self.window.name, self.window.contour_img)
                else:
                    cv.imshow(self.window.name, self.window.og_img)

            if self.window.refresh_on_next:
                self.window.show_img()

                st, lines = self.window.extract_stats()
                im, st_im = put_textbox_on_img(self.window.og_img, lines)
                self.stats_img = st_im
                self.change_stats_image()
                self.window.update_contour_img(False)
                self.window.refresh_on_next = False

            key = cv.waitKey(100)

            # Breaks infinite loop if SPACE is pressed OR OpenCV window is closed
            if key == 32 or cv.getWindowProperty(self.window.name, cv.WND_PROP_VISIBLE) < 1:
                self.op = False
                break

            await asyncio.sleep(0)
        cv.destroyAllWindows()

        # Awaiting end
        await asyncio.sleep(0)
        print('Image Coroutine is done')

    def coroutine_controls(self):
        """Tkinter controls & stats window coroutine"""
        root = tk.Tk()
        self.tk = root

        image_frame = tk.Frame(root)
        image_frame.pack(side='left', fill='both')

        if self.stats_img is not None:
            im = Image.fromarray(self.stats_img)
        else:
            im = Image.fromarray(np.zeros((200, 200)))

        photo = ImageTk.PhotoImage(image=im)

        image_label = tk.Label(image_frame, image=photo)
        self.tk_photo = image_label
        image_label.pack()

        controls_frame = tk.Frame(root)
        controls_frame.pack(fill='both', expand=True)

        rad_btn_frame = tk.Frame(controls_frame)
        rad_btn_val = tk.BooleanVar()
        rad_btn_frame.pack()

        R1 = tk.Radiobutton(rad_btn_frame, text="SELECT", variable=rad_btn_val, value=True,
                            command=placeholder_func)
        R2 = tk.Radiobutton(rad_btn_frame, text="EDIT", variable=rad_btn_val, value=False,
                            command=placeholder_func)

        R1.pack(anchor=tk.W, side=tk.LEFT)
        R2.pack(anchor=tk.W, side=tk.LEFT)

        checkButtons = {}

        clusters = [Cluster("c1", (40, 150, 40)), Cluster("c2", (110, 150, 40)), Cluster(

            "c3", (110, 150, 40)), Cluster("c4", (110, 150, 40))]
        cb_vals = [tk.BooleanVar(value=True) for i in clusters]

        def checkbox_update():
            for i, val in enumerate(cb_vals):
                self.window.clusters[i].checked = val.get()
            print([i.checked for i in self.window.clusters])
            self.window.update_contour_img()

        for i, cluster in enumerate(clusters):
            checkButtons.update({cluster.name: tk.Checkbutton(controls_frame, text=cluster.name,
                                variable=cb_vals[i], onvalue=1, offvalue=0, width=1, command=checkbox_update)})

        for io in checkButtons.values():
            io.pack()

        btn_frame = tk.Frame(controls_frame)
        btn_frame.pack(fill='y', expand=True)

        stat_btn_frame = tk.Frame(btn_frame)
        stat_btn_frame.pack(fill='y', expand=False)

        button1 = tk.Button(stat_btn_frame, text="Show Stats",
                            command=placeholder_func)
        button2 = tk.Button(stat_btn_frame, text="Save Stats",
                            command=placeholder_func)
        button3 = tk.Button(btn_frame, text="Dis-/Enable Contours",
                            command=self.window.toggle_contours)
        button1.pack(side=tk.LEFT)
        button2.pack(side=tk.LEFT)
        button3.pack()

        root.mainloop()

    def change_stats_image(self):
        im = Image.fromarray(self.stats_img)
        photo = ImageTk.PhotoImage(image=im)

        self.tk_photo.configure(image=photo)
        self.tk_photo.image = photo

    async def main(self):
        # create the coroutines
        coroutine1 = self.coroutine_zmq()
        coroutine2 = self.coroutine_image()
        # coroutine3 = self.coroutine_controls()

        # schedule the coroutine to run in the background
        task1 = asyncio.create_task(coroutine1)
        task2 = asyncio.create_task(coroutine2)
        # task3 = asyncio.create_task(coroutine3)

        # simulate continue on with other things
        await task1
        await task2
        # await task3


def placeholder_func(*args):
    print("placeholder_func")
    print(args)


def put_text(img: np.ndarray,
             text: str,
             org: "tuple(int, int)",
             font=cv.FONT_HERSHEY_PLAIN,
             fontScale: int = 1,
             colors: "tuple(tuple(int, int, int), tuple(int, int, int))" = (
                 (0, 0, 0), (255, 255, 255)),
             thickness: int = 3) -> np.ndarray:
    im = cv.putText(img.copy(), text, org, font, fontScale,
                    colors[0], thickness, cv.LINE_AA)
    im = cv.putText(im, text, org, font, fontScale,
                    colors[1], 1, cv.LINE_AA)
    return im


def put_textbox_on_img(img, lines: "list[str]", start_point: "tuple(int, int)" = (0, 0), width=260):
    image = img.copy()

    offset_y, offset_x = 30, 20
    line_h = 40

    point = (start_point[0], start_point[1])
    end_point = (point[0] + len(lines) * line_h, point[1] + width)

    # Ensuring that the textbox is visible
    diff = (img.shape[1] - end_point[0], img.shape[0] - end_point[1])
    if diff[0] < 0:
        point = (point[0] + diff[0], point[1])
        end_point = (end_point[0] + diff[0], end_point[1])
    if diff[1] < 0:
        point = (point[0], point[1] + diff[1])
        end_point = (end_point[0], end_point[1] + diff[1])

    # Creating box
    box_end_point = (end_point[0], end_point[1] + offset_y)
    image = cv.rectangle(
        image, point, box_end_point, (0, 0, 0), 3)
    image = cv.rectangle(
        image, point, box_end_point, (240, 240, 240), -1)

    # Inserting lines of text
    point = (point[0] + offset_x, point[1] + offset_y)
    for line in lines:
        image = put_text(image, str(line), point, font=cv.FONT_HERSHEY_PLAIN)
        point = (point[0], point[1] + line_h)
    return image, image[start_point[0]:end_point[0], start_point[1]:end_point[1]]


def find_object_for_point(point: "tuple[int,int]", clusters: "list[Cluster]", disable=False) -> ObjectParams or None:
    for cluster in clusters:
        if not cluster.checked:
            continue

        for j, contour in enumerate(cluster.contours):

            # If point is on or inside the contour
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:

                # Move to disabled list if disabling is turned on
                if disable:
                    cluster.disable_contour(j)
                    return ObjectParams(cluster.name, len(cluster.disabled_contours) - 1, cv.contourArea(contour), centroid_for_contour(contour), True)

                return ObjectParams(cluster.name, j, cv.contourArea(contour), centroid_for_contour(contour), False)

        for j, contour in enumerate(cluster.disabled_contours):

            # If point is on or inside the contour
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:

                # Move to enabled list if disabling is turned on
                if disable:
                    cluster.enable_contour(j)
                    return ObjectParams(cluster.name, len(cluster.contours) - 1, cv.contourArea(contour), centroid_for_contour(contour), False)

                return ObjectParams(cluster.name, j, cv.contourArea(contour), centroid_for_contour(contour), True)
    return None


def centroid_for_contour(contour):
    M = cv.moments(contour)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    return (cx, cy)


if __name__ == "__main__":
    # run the asyncio program
    a = BigTing()

    asyncio.run(a.main())
