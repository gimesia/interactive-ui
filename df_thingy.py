import asyncio
import csv
from datetime import datetime
import os
from pathlib import Path
import threading
import tkinter as tk
import cv2 as cv
import numpy as np
import pandas as pd
from zmq import Context, Socket
import zmq
from PIL import Image, ImageTk
from stardist.models import StarDist2D
from stardist.nms import non_maximum_suppression
from stardist.geometry import dist_to_coord
from colour_deconvolution import ColourDeconvolution

# Creating constants
VERBOSE = True
COLUMNS = ["Cluster", "Disabled", "Contour", "Center", "Area"]

# ZMQ Constants
PING = "PING"
THRESHOLD_INFO = "THRESHOLD_INFO"
INP_IMAGE = "INP_IMAGE"
QUERY_CONTOUR = "QUERY_CONTOUR"
REQ_RECEIVED = "REQUEST_RECEIVED"
DONE = "DONE"
FAILED = "FAILED"
ALIVE = "ALIVE"
EXIT = "EXIT"
UNKNOWN = "UNKNOWN"
ZMQ_SERVERNAME = "tcp://*:5560"

# Output Constants
OUTPUT_DIR = os.path.join(str(Path.home() / "Downloads"), "rescore_ui_output")
TODAY = datetime.now().strftime("%Y-%m-%d")
TODAY_DIR = os.path.join(OUTPUT_DIR, TODAY)

# Contour prediction constants
REFERENCE_UMPP = 0.125
COLOUR_DECONVOLUTION = ColourDeconvolution([
    [0.650, 0.704, 0.286],
    [0.268, 0.570, 0.776],
    # [0, 0, 0]
])
MODEL = StarDist2D.from_pretrained("2D_versatile_fluo")

MAINFRAME = pd.DataFrame({
    "Cluster": [],
    "Disabled": pd.Series([], dtype=bool),
    "Contour": [],
    "Center": [],
    "Area": []
})


def predict_contours(
        stardist_model: StarDist2D,
        image: np.ndarray,
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.5,
        predict_kwargs: dict = {},
        nms_kwargs: dict = {}
) -> np.ndarray:
    """Predict contours from images by StarDist2D models."""
    probabilities, distances = stardist_model.predict(
        image,
        **predict_kwargs
    )
    star_centre_points, probabilities, distances = non_maximum_suppression(
        distances,
        probabilities,
        grid=stardist_model.config.grid,
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
        **nms_kwargs
    )
    coordinates = dist_to_coord(
        distances,
        star_centre_points,
    )
    contours = np.transpose(  # Transforming to list of list of points format
        coordinates,
        [0, 2, 1]
    )
    contours = np.take(  # transforming from height, width to x, y coordinates.
        contours,
        [1, 0],
        axis=2
    )
    return contours

class Cluster():
    """Class for a class of identified objects"""
    global mainframe

    def __init__(self, clustername: str, color: "tuple(int, int, int)", thickness: int):
        self.name = clustername
        self.color = color
        self.thickness = thickness


class ImageWindow():
    def __init__(self, name="Interactive UI"):
        # CREATE CLUSTERS
        self.clusters: list[Cluster] = [
            Cluster("Positive", (107, 76, 254), 1),
            Cluster("Negative", (250, 106, 17), 1),
        ]
        self.name = name
        self.df = MAINFRAME.copy()
        self.sample_df = MAINFRAME.copy()
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.show_contours = True
        self.refresh_on_next = False
        self.edit = True
        self.stats = None
        self.stats_img = None
        

        self.set_base_image(cv.imread("img/proto_img.tiff"))
        self.seg()
        self.update_contours()
        
        cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE)
        # self.update_contour_img()

        # Mouse event callbacks
        # cv.setMouseCallback(self.name, self.on_mouse_event)

        # for cl in self.clusters:
        #     cl.sort_contours()
        # self.data = self.extract_dataframe_data()
       
    def set_base_image(self, img: np.ndarray) -> None:
        """Set new image as reference in the object

        Args:
            img (np.ndarray): 
        """
        # TODO Resize!
        self.og_img = img.copy()
        self.contour_img = img.copy()
        
    def show_img(self) -> None:
        if self.show_contours:
            cv.imshow(self.name, self.contour_img)
        else:
            cv.imshow(self.name, self.og_img)


    def seg(self):
        """TODO: Move this step outside this file, it should get only the contours"""
        img = self.og_img.copy()
        self.sample_df = MAINFRAME.copy()

        # Displaying image without contours temporarily
        cv.imshow(self.name, img)
        
        # Color deconvolution
        concentration_maps = COLOUR_DECONVOLUTION.get_concentration(img, normalisation="scale")        
        for stain_index, cluster in enumerate(self.clusters):
            conts = list(
                np.rint(
                    predict_contours(
                        MODEL,
                        concentration_maps[..., stain_index],
                        prob_thresh=0.3,
                        nms_thresh=0.1
                    )
                ).astype(int)
            )
            
            # Create series from segmented objects
            conts_series = pd.Series(conts)
            clusternames_series = pd.Series(
                np.full_like(conts_series, cluster.name))
            area_series = pd.Series(
                list(map(lambda cnt: cv.contourArea(cnt), conts)))
            centroid_series = pd.Series(
                list(map(lambda cnt: centroid(cnt), conts)))

            # Create df from series
            df = pd.DataFrame({
                "Cluster": clusternames_series,
                "Disabled": pd.Series(np.zeros_like(conts_series, dtype=bool)),
                "Selected": pd.Series(np.zeros_like(conts_series, dtype=bool)),
                "Contour": conts_series,
                "Center": centroid_series,
                "Area": area_series
            })

            # Adding rows into object property
            self.df = pd.concat([self.df, df], ignore_index=True)

        self.sample_df = self.df.copy()
        return
    
    def update_contours(self):
        """Updates the contours on the image based on the 'sample_df' object variable
        """
        im = self.og_img.copy()
        
        for i in self.sample_df.iterrows():
            cls = self.clusters
            row = i[1]
            mapped = list(map(lambda x: x.name, cls))
            print(row["Cluster"])
            color = cls[mapped.index(row["Cluster"])].color
            print(row["Contour"])
            print(type(row["Contour"]))
            cv.drawContours(im, [row["Contour"]], -1, color, 1, cv.LINE_AA)
        
        self.contour_img = im
        self.show_img()        



class BigTing():
    def __init__(self):
        self.context: Context = Context()
        self.socket: Socket = self.context.socket(zmq.PAIR)
        self.window = ImageWindow()
        self.is_open = True
        self.tk: tk.Tk = None
        self.tk_photo: np.ndarray = None

        self.socket.bind(ZMQ_SERVERNAME)

    def handle_message(self, msg: str):
        print(f"Command: {msg}, redirecting accordingly")
        if msg == PING:
            self.pong()
        elif msg == INP_IMAGE:
            self.receive_image()
        elif msg == QUERY_CONTOUR:
            self.send_contours()
        elif msg == EXIT:
            self.confirm_exit(True)
        else:
            self.socket.send_string(f"{msg}")

    def pong(self):  # Answer to ping
        self.socket.send_string(ALIVE)

    def confirm_req(self):  # Sends confirmation of received request
        print("Sending confirmation of received request")
        self.socket.send_string(REQ_RECEIVED)

    def confirm_req_complete(self):  # Sends confirmation of received request
        print("Sending confirmation")
        self.socket.send_string(DONE)

    def confirm_req_failed(self):  # Sends receit of failed request
        print("Sending fail receipt")
        self.socket.send_string(FAILED)

    def confirm_exit(self, close=True):  # Confirms EXIT command
        print("Sending EXIT confirmation")
        if close:
            self.is_open = False
        self.socket.send_string("q")

    def receive_image(self):  # Reception of image from socket
        global now
        now = datetime.now().strftime("%H-%M-%S")

        print("Receiving image")
        self.confirm_req()
        message = self.socket.recv_pyobj()
        try:
            message = cv.cvtColor(message, cv.COLOR_BGR2RGB)
            self.window.set_base_image(message)
            self.confirm_req_complete()
        except:
            self.confirm_req_failed()
        self.window.update_contour_img(segment=True)
        self.window.refresh_on_next = True

    def send_contours(self):
        contours = [cl.contours for cl in self.window.clusters]
        self.socket.send_pyobj(contours)

    async def coroutine_zmq(self):
        """ZeroMQ communication async coroutine"""
        print("ZMQ Coroutine is running")
        i = 0
        while self.is_open:
            if VERBOSE:
                print(f"zmq iter: {i}")
            i += 1
            message = None
            if self.socket.poll(100, zmq.POLLIN):
                message = self.socket.recv_string()
                self.handle_message(message)
            await asyncio.sleep(0)

        self.socket.close()
        self.tk.quit()
        await asyncio.sleep(0)
        print("ZMQ Coroutine is done")

    async def coroutine_image(self):
        """OpenCV image display coroutine"""
        print("Image Coroutine is running")

        # Starting tkinter bg thread
        thread = threading.Thread(target=self.coroutine_controls)
        thread.start()

        # Starting opencv loop
        j = 0
        while self.is_open:
            if VERBOSE:
                print(f"image iter: {j}")
            j += 1

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
                    st_im = textbox(["lorem ipsum", "ipsum lorem"])  # lines)
                    self.window.stats_img = st_im

                self.change_stats_image()
                self.window.update_contour_img(False)
                self.window.refresh_on_next = False

            key = cv.waitKey(250)

            # Breaks infinite loop if SPACE is pressed OR OpenCV window is closed
            if key == 32 or cv.getWindowProperty(self.window.name, cv.WND_PROP_VISIBLE) < 1:
                self.is_open = False
                break

            await asyncio.sleep(0)

        cv.destroyAllWindows()
        self.tk.quit()

        # Awaiting end
        await asyncio.sleep(0)
        print("Image Coroutine is done")

    def coroutine_controls(self):
        """Tkinter controls & stats window coroutine"""
        root = tk.Tk()
        root.title("Controls & Statistics")
        self.tk = root

        image_frame = tk.Frame(root)
        image_frame.pack(side="left", fill="both", padx=12, pady=12)

        if self.window.stats_img is not None:
            im = Image.fromarray(self.windwo.stats_img)
        else:
            im = Image.fromarray(np.zeros((200, 200)))

        photo = ImageTk.PhotoImage(image=im)

        image_label = tk.Label(image_frame, image=photo)
        self.tk_photo = image_label
        image_label.pack()

        controls_frame = tk.Frame(root)
        controls_frame.pack(fill="both", expand=True, padx=(0, 25), pady=25)

        rad_btn_frame = tk.Frame(controls_frame)
        rad_btn_val = tk.BooleanVar()
        rad_btn_frame.pack()

        R1 = tk.Radiobutton(rad_btn_frame, text="SELECT", variable=rad_btn_val, value=True)
        R2 = tk.Radiobutton(rad_btn_frame, text="EDIT", variable=rad_btn_val, value=False)
        R1.pack(anchor=tk.W, side=tk.LEFT)
        R2.pack(anchor=tk.W, side=tk.LEFT)

        btn_frame = tk.Frame(controls_frame)
        btn_frame.pack(fill="y", expand=True)

        def a():
            print(self.window.sample_df)

        button1 = tk.Button(btn_frame, text="Save STATS", command=a)
        button2 = tk.Button(btn_frame, text="Save DATA")
        button3 = tk.Button(btn_frame, text="Dis-/Enable Contours")
        button1.pack()
        button2.pack()
        button3.pack()

        def on_closing():
            self.is_open = False

        root.protocol("WM_DELETE_WINDOW", on_closing)
        root.mainloop()

    def change_stats_image(self):
        im = Image.fromarray(np.zeros((200,400)))#self.window.stats_img)
        photo = ImageTk.PhotoImage(image=im)
        self.tk_photo.configure(image=photo)
        self.tk_photo.image = photo

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
        
    def run(self):
        asyncio.run(self.main())


def centroid(contour: np.ndarray) -> "tuple(int, int)":
    """Calculates the centroid for a contour

    Args:
        contour (np.ndarray): polygon

    Returns:
        tuple(int, int): center point
    """
    M = cv.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)

        
if __name__ == "__main__":
    # run the asyncio program
    a = BigTing()
    a.run()

