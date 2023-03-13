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

ZMQ_SERVERNAME = "tcp://*:5560"


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
                    "center": centroid_for_contour(contour)
                }})
        feat_coll = {
            "type": "FeatureCollection",
            "features": features,
            "properties": {
                "cluster": self.name,
                "index": i
            }            
        }
        return feat_coll


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
        self.th_c = 60
        self.blocksize = 251
        self.show_contours = True
        self.refresh_on_next = False
        self.edit = True
        self.stats = None
        self.stats_img = None

        cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE | cv.WINDOW_KEEPRATIO)


        self.set_base_image(cv.imread("cells_6.jpg"))
        self.show_img()
        self.update_contour_img()
        
        # Mouse event callbacks
        cv.setMouseCallback(self.name, self.on_mouse_event)
        # self.create_ui_controls()

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

    def ath(self):
        """TODO: BOTI CHANGE THIS!!!

        Returns:
            clusters: Classified clusters with the stored contours 
        """
        img = self.og_img.copy()
        gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        th = cv.adaptiveThreshold(
            gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, self.blocksize, self.th_c)

        contours, h = cv.findContours(
            th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        contours_map = list((map(lambda x: cv.contourArea(x), contours)))

        if not len(contours_map):
            return contours

        # Classification
        mx, mn = max(contours_map), 100#min(contours_map)
        diff = (mx - mn) / 4

        for cl in self.clusters:
            cl.contours = []
            cl.disabled_contours = []

        for i, cont in enumerate(contours):
            # Filter out small specs 
            if contours_map[i] < 100:
                continue
            
            if contours_map[i] < mn + diff:
                self.clusters[0].contours.append(cont)
            elif contours_map[i] < mn + 2 * diff:
                self.clusters[1].contours.append(cont)
            elif contours_map[i] < mn + 3 * diff:
                self.clusters[2].contours.append(cont)
            else:
                self.clusters[3].contours.append(cont)
        return self.clusters

    def toggle_select_mode(self):
        print("SELECT MODE")
        self.edit = False
        self.refresh_on_next = True

    def toggle_edit_mode(self):
        print("EDIT MODE")
        self.edit = True
        self.refresh_on_next = True

    def save_data(self, *args):
        # TODO!
        print("save_data:")
        print(args)
        pass

    def set_block_size(self, val, *args):
        """!!!DEPRECATED!!!"""
        print(f"set_block_size: {val}")
        if val % 2 == 0:
            if val > 1:
                val += 1
            else:
                val = 3
        self.blocksize = val
        self.update_contour_img()

    def set_c_value(self, val, *args):
        """!!!DEPRECATED!!!"""
        print(f"set_c_value: {val}")
        self.th_c = val
        self.update_contour_img()

    def create_ui_controls(self):
        """!!!DEPRECATED!!!"""
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
            cv.drawContours(im, cl.contours, -1, cl.color, 2)
            cv.drawContours(im, cl.disabled_contours, -1, (100, 100, 100), 2)
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

        indexes1 = list(map(lambda cl: cl.name, self.clusters))

        if not all:
            frame = {'Count': np.zeros(
                len(lengths),), 'Percentage': np.zeros(len(lengths),)}
        else:
            frame = {'Count': pd.Series(lengths, index=indexes1),
                     'Percentage': pd.Series((lengths / enabled) * 100, index=indexes1)}

        count = [enabled, disabled, all]
        percentage = [100 * enabled / all, 100 * disabled / all, int(100)]
        indexes2 = ['enabled', 'disabled', 'all', ]

        df1 = pd.DataFrame(frame, index=indexes1)
        df2 = pd.DataFrame({'Count': count,
                            'Percentage': percentage},
                           index=indexes2)
        res = pd.concat(objs=[df1, df2])

        self.stats = res

        lines = []
        for index, row in res.iterrows():
            lines.append(
                f"{index}: {int(row['Count'])} ({round(row['Percentage'], 2) if bool(row['Count']) else '0'}%)"
            )

        return res, lines

    def claster_by_name(self, cluster_name: str) -> Cluster:
        index = list(map(lambda x: x.name, self.clusters)).index(cluster_name)
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

        if self.edit:
            if event == 4:
                """if flags == 17:
                    # shift
                    obj = find_object_for_point(point, self.clusters, False)
                    self.change_cluster(obj, backwards=True)
                elif flags == 9:
                    # ctrl
                    obj = find_object_for_point(point, self.clusters, False)
                    self.change_cluster(obj, backwards=False)"""
                obj = find_object_for_point(point, self.clusters, True)

            elif event == 2:
                (print("Rescore object"))
                obj = find_object_for_point(point, self.clusters, False)
                if obj is not None:
                    self.change_cluster(obj, backwards=False)

            self.refresh_on_next = True
        else:
            if event == 4: 
                (print("Select object"))
                obj = find_object_for_point(point, self.clusters, False)
                obj_stats_img= textbox(obj.__str__().split("; "))
                self.stats_img = obj_stats_img
                self.refresh_on_next = True


class BigTing():
    def __init__(self):
        self.context: Context = Context()
        self.sock: Socket = self.context.socket(zmq.PAIR)
        self.sock.bind(ZMQ_SERVERNAME)
        self.op = True
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
        elif msg == QUERY_CONTOUR:
            self.sock.send_pyobj(f"")
        else:
            self.sock.send_string(f"{msg}")

    def pong(self):  # Answer to ping
        self.sock.send_string(ALIVE)

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
            message = cv.cvtColor(message, cv.COLOR_BGR2RGB)
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
        self.tk.quit()
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

                if self.window.edit:
                    st, lines = self.window.extract_stats()
                    st_im = textbox(lines)
                    self.window.stats_img = st_im
                    

                self.change_stats_image()
                self.window.update_contour_img(False)
                self.window.refresh_on_next = False

            key = cv.waitKey(250)

            # Breaks infinite loop if SPACE is pressed OR OpenCV window is closed
            if key == 32 or cv.getWindowProperty(self.window.name,
                                                 cv.WND_PROP_VISIBLE) < 1:
                self.op = False
                break

            await asyncio.sleep(0)

        cv.destroyAllWindows()
        self.tk.quit()

        # Awaiting end
        await asyncio.sleep(0)
        print('Image Coroutine is done')

    def coroutine_controls(self):
        """Tkinter controls & stats window coroutine"""
        root = tk.Tk()
        self.tk = root

        image_frame = tk.Frame(root)
        image_frame.pack(side='left', fill='both')

        if self.window.stats_img is not None:
            im = Image.fromarray(self.windwo.stats_img)
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

        R1 = tk.Radiobutton(rad_btn_frame, text="SELECT", variable=rad_btn_val,
                            value=True, command=self.window.toggle_select_mode)
        R2 = tk.Radiobutton(rad_btn_frame, text="EDIT", variable=rad_btn_val,
                            value=False, command=self.window.toggle_edit_mode)

        R1.pack(anchor=tk.W, side=tk.LEFT)
        R2.pack(anchor=tk.W, side=tk.LEFT)

        check_buttons = {}

        clusters = self.window.clusters
        cb_vals = [tk.BooleanVar(value=True) for i in clusters]

        def checkbox_update():
            for i, val in enumerate(cb_vals):
                self.window.clusters[i].checked = val.get()
            self.window.refresh_on_next = True

        for i, cluster in enumerate(clusters):
            check_buttons.update({
                cluster.name: tk.Checkbutton(
                    controls_frame,
                    text=cluster.name,
                    variable=cb_vals[i],
                    onvalue=1,
                    offvalue=0,
                    width=1,
                    command=checkbox_update
                )
            })

        for io in check_buttons.values():
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

        def on_closing():
            self.op = False

        root.protocol("WM_DELETE_WINDOW", on_closing)

        root.mainloop()

    def change_stats_image(self):
        im = Image.fromarray(self.window.stats_img)
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
                font=cv.FONT_HERSHEY_SIMPLEX,
                fontScale: int = 1,
                colors: "tuple(tuple(int, int, int), tuple(int, int, int))" = (
                    (0, 0, 0), (255, 255, 255)),
                thickness: int = 3) -> np.ndarray:
    im = cv.putText(img, text, org, font, fontScale,
                    colors[0], thickness, cv.LINE_AA)
    im = cv.putText(im, text, org, font, fontScale,
                    colors[1], 1, cv.LINE_AA)
    return im


def find_object_for_point(point: "tuple[int,int]",clusters: "list[Cluster]",disable=False)-> ObjectParams or None:
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


def textbox(lines: "list[str]"):
    LINE_H = 40
    OFFSET_X = 10
    OFFSET_Y = 10
    CHAR_W = 18

    end_h = len(lines) * LINE_H + OFFSET_Y
    end_w = max(list(map(lambda x: len(x), lines)))

    im = np.zeros((end_h, end_w * CHAR_W, 3), dtype=np.uint8)
    cv.rectangle(im, (0, 0), (end_w * CHAR_W, end_h), (240, 240, 240), -1)

    point = (OFFSET_X, LINE_H - OFFSET_Y)
    for line in lines:

        put_text(im, line, point)
        point = (point[0], point[1] + LINE_H)
            
    return im


if __name__ == "__main__":
    # run the asyncio program
    a = BigTing()

    asyncio.run(a.main())
