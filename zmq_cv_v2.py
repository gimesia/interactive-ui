
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

# Contour prediction parameters
REFERENCE_UMPP = 0.125
COLOUR_DECONVOLUTION = ColourDeconvolution([
    [0.650, 0.704, 0.286],
    [0.268, 0.570, 0.776],
    # [0, 0, 0]
])
MODEL = StarDist2D.from_pretrained("2D_versatile_fluo")

# Global pd.DataFrame variable
mainframe = pd.DataFrame({
    "Cluster": [],
    "Disabled": pd.Series([], dtype=bool),
    "Contour": [],
    "Center": [],
    "Area": []
})


# Dynamic functions for saving results
def now(): return datetime.now().strftime("%H-%M-%S")
def data_destination_path_raw(): return f"{TODAY_DIR}/{now()}_raw-data"
def data_destination_path_supervised(
): return f"{TODAY_DIR}/{now()}_supervised-data"
def stats_destination_path(): return f"{TODAY_DIR}/{now()}_stats"


# Creating output files
# TODO: further isolation of saved data of different quants
try:
    os.listdir(OUTPUT_DIR)
except:
    os.mkdir(OUTPUT_DIR)
try:
    os.listdir(TODAY_DIR)
except:
    os.mkdir(TODAY_DIR)


class Cluster():
    """Class for a class of identified objects"""
    global mainframe

    def __init__(self, clustername: str, color: "tuple(int, int, int)", thickness: int):
        self.name = clustername
        self.color = color
        self.thickness = thickness

    def get_data(self, df: pd.DataFrame, *args) -> pd.DataFrame:
        return df[(df["Cluster"] == self.name)]

    def contours(self, *args) -> pd.Series:
        return self.get_data()["Contour"]

    def sort_contours(self):
        return

    def contour_count(self) -> int:
        pass

    def disable_contour(self, index: int) -> None:
        pass

    def enable_contour(self, index: int) -> None:
        pass

    def to_geojson(self, disableds=False):
        return


class ImageWindow():
    def __init__(self, name="Interactive UI"):
        # CREATE CLUSTERS
        self.clusters: list[Cluster] = [
            Cluster("Positive", (107, 76, 254), 1),
            Cluster("Negative", (250, 106, 17), 1),
        ]
        self.name = name
        self.raw_df = mainframe.copy()
        self.edit_df = mainframe.copy()
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.show_contours = True
        self.refresh_on_next = False
        self.edit = True
        self.stats = None
        self.stats_img = None

        cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE)
        self.set_base_image(cv.imread("img/proto_img.tiff"))
        self.update_contour_img()

        # Mouse event callbacks
        cv.setMouseCallback(self.name, self.on_mouse_event)

        for cl in self.clusters:
            cl.sort_contours()
        self.data = self.extract_dataframe_data()

    def set_base_image(self, img: np.ndarray) -> None:
        # TODO Resize!
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

    def segmentation(self):
        global MODEL
        img = self.og_img.copy()

        # Emptying dataframe
        self.raw_df = pd.DataFrame({
            "Cluster": [],
            "Disabled": pd.Series([], dtype=bool),
            "Contour": [],
            "Center": [],
            "Area": []
        })

        # Displaying image without contours temporarily
        cv.imshow(self.name, img)

        # Color deconvolution
        concentration_maps = COLOUR_DECONVOLUTION.get_concentration(
            img,
            normalisation="scale"
        )
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
                list(map(lambda cnt: centroid_for_contour(cnt), conts)))

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
            self.raw_df = pd.concat([self.raw_df, df])

        # Storing the editable dataframe
        self.edit_df = self.raw_df.copy()
        return

    def toggle_select_mode(self):
        print("SELECT MODE")
        self.edit = False
        self.refresh_on_next = True

    def toggle_edit_mode(self):
        print("EDIT MODE")
        self.edit = True
        self.refresh_on_next = True

    def extract_geojson_data(self):
        """TODO"""
        return

    def extract_dataframe_data(self) -> pd.DataFrame:
        """TODO"""
        return

    def save_data(self, *args):
        """TODO"""
        global mainframe
        data = self.edit_df
        print(mainframe)
        return

    def extract_stats(self, *args):
        """TODO"""
        return

    def save_stats(self, *args):
        """TODO"""
        return

    def update_contour_img(self, segment=True) -> None:
        """Updates the displayed image on the window

        Args:
            segment (bool, optional): run segmentation before updating. Defaults to True.
        """
        if segment:
            self.segmentation()

        # Draw contours for every cluster
        for cl in self.clusters:
            cluster_df = cl.get_data(self.edit_df)
            cnts = (cluster_df["Contour"])
            for i, cnt in enumerate(cnts):
                disabled = cluster_df["Disabled"][i]
                selected = cluster_df["Selected"][i]
                color = cl.color
                if selected:
                    color = (0, 210, 10)
                elif disabled:
                    color = (10, 10, 10)
                cv.drawContours(self.contour_img, [cnt],
                                -1, color, cl.thickness)

        cv.imshow(self.name, self.contour_img)
        return

    def claster_by_name(self, cluster_name: str) -> Cluster:
        index = list(map(lambda x: x.name, self.clusters)).index(cluster_name)
        return self.clusters[index]

    def change_cluster(self, backwards=False):
        pass

    def on_mouse_event(self, event, x: int, y: int, flags, *args) -> None:
        if self.og_img is None or self.contour_img is None or not event:
            return
        point = (x, y)
        print(point)

        # Edit Mode
        if self.edit:
            if event == 4:
                print("Dis-/Enable object")
                obj = self.toggle_object(point)
            elif event == 2:
                print("Rescore object")
                obj = self.rescore_object(point)
            else:
                return
            self.refresh_on_next = True

        # Select Mode
        else:
            if event == 4:
                print("Select object")
                self.edit_df["Selected"] = False
                self.select_object(point)
                self.refresh_on_next = True

    def select_object(self, point, multiple=False):
        """Changes 'Selected' property """
        condition = self.edit_df["Contours"].apply(
            lambda x: cv.pointPolygonTest(x[0], point, False) >= 0
        )
        filtered = self.edit_df[condition]
        self.edit_df.loc[filtered.index, "Selected"] = True
        return

    def disable_object(self, point):
        """TODO"""
        condition = self.edit_df["Contours"].apply(
            lambda x: cv.pointPolygonTest(x[0], point, False) >= 0
        )
        filtered = self.edit_df[condition]
        pass

    def rescore_object(self, point):
        """TODO"""
        pass


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
        self.window.segmentation()
        self.window.update_contour_img()
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
        start_new_thread(self.coroutine_controls)
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

        R1 = tk.Radiobutton(rad_btn_frame, text="SELECT", variable=rad_btn_val, value=True,
                            command=self.window.toggle_select_mode)
        R2 = tk.Radiobutton(rad_btn_frame, text="EDIT", variable=rad_btn_val, value=False,
                            command=self.window.toggle_edit_mode)

        R1.pack(anchor=tk.W, side=tk.LEFT)
        R2.pack(anchor=tk.W, side=tk.LEFT)

        check_buttons = {}

        clusters = self.window.clusters
        cb_vals = [tk.BooleanVar(value=True) for i in range(2)]

        def checkbox_update():
            return
            for i, val in enumerate(cb_vals):
                self.window.clusters[i].checked = val.get()
            self.window.refresh_on_next = True

        # for i, cluster in enumerate(clusters):
        #     check_buttons.update({
        #         cluster.name: tk.Checkbutton(controls_frame, text=cluster.name, variable=cb_vals[i], onvalue=1, offvalue=0, width=10, command=placeholder_func)
        #     })

        for io in check_buttons.values():
            io.pack()

        btn_frame = tk.Frame(controls_frame)
        btn_frame.pack(fill="y", expand=True)

        button1 = tk.Button(btn_frame, text="Save STATS",
                            command=self.window.save_stats)
        button2 = tk.Button(btn_frame, text="Save DATA",
                            command=self.window.save_data)
        button3 = tk.Button(btn_frame, text="Dis-/Enable Contours",
                            command=self.window.toggle_contours)
        button1.pack()
        button2.pack()
        button3.pack()

        def on_closing():
            self.is_open = False

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


def find_object_for_point(point: "tuple(int, int)", df: pd.DataFrame, multi_select=False) -> pd.DataFrame:

    series = df["Contour"]
    p_test = np.array([cv.pointPolygonTest(x, point, False) for x in series])
    print(p_test)
    print(np.argwhere(p_test > -1).shape)
    indexes = np.argwhere(p_test > -1)

    if not np.any(p_test):
        return pd.DataFrame({})

    hits = df.loc[indexes.squeeze()]
    a = hits["Area"].min()
    min_area = hits.loc[hits["Area"] == a]

    print(f"hits:\n{hits}")
    print(a)
    print(f"min_area:\n{min_area}")

    if multi_select:
        return hits

    return min_area


def centroid_for_contour(contour: np.ndarray) -> "tuple(int, int)":
    """Calculates the center of a contour

    Args:
        contour (np.ndarray): polygon

    Returns:
        tuple(int, int): center point
    """
    M = cv.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def put_text(img: np.ndarray,
             text: str,
             org: "tuple(int, int)",
             font=cv.FONT_HERSHEY_SIMPLEX,
             fontScale: int = 1,
             colors: "tuple(tuple(int, int, int), tuple(int, int, int))"
             = ((0, 0, 0), (255, 255, 255)),
             thickness: int = 3) -> np.ndarray:
    im = cv.putText(img, text, org, font, fontScale,
                    colors[0], thickness, cv.LINE_AA)
    im = cv.putText(im, text, org, font, fontScale,
                    colors[1], 1, cv.LINE_AA)
    return im


def textbox(lines: "list[str]"):
    LINE_H = 40
    OFFSET_X = 10
    OFFSET_Y = 10
    CHAR_W = 18

    # Getting end coords
    end_h = len(lines) * LINE_H + OFFSET_Y
    end_w = max(list(map(lambda x: len(x), lines)))

    # Creating container box
    im = np.zeros((end_h, end_w * CHAR_W, 3), dtype=np.uint8)
    cv.rectangle(im, (0, 0), (end_w * CHAR_W, end_h), (240, 240, 240), -1)

    # Initializing starting point
    point = (OFFSET_X, LINE_H - OFFSET_Y)

    # Writing lines
    for line in lines:
        put_text(im, line, point)
        point = (point[0], point[1] + LINE_H)
    return im


def predict_contours(
        stardist_model: StarDist2D,
        image: np.ndarray,
        prob_thresh: float = 0.5,
        nms_thresh: float = 0.5,
        predict_kwargs: dict = {},
        nms_kwargs: dict = {}
) -> np.ndarray:
    """
    Predict contours from images by StarDist2D models.

    Parameters
    ----------
    stardist_model : StarDist2D
        The the trained stardist 2D model which we want to use for prediction.
    image : ndarray
        The image on which we want to predict.
    prob_thresh : float, optional
        Probability threshold for non-maximum suppression selection of
        the best contours.
        The default is 0.5 .
    nms_thresh : TYPE, optional
        Non-maximum suppression threshold for selection of
        the best contours.
        The default is 0.5 .
    predict_kwargs : dict, optional
        Keyword arguments for predict method of the stardist 2D model.
        The default is {}.
    nms_kwargs : dict, optional
        Keyword arguments for non_maximum_suppression function.
        The default is {}.

    Returns
    -------
    contours : ndarray
        Array of predicted contours. The contours are arrays of x, y point
        coordinate tuples.

    """
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


def start_new_thread(fc):
    thread = threading.Thread(target=fc)
    thread.start()


if __name__ == "__main__":
    # run the asyncio program
    a = BigTing()

    asyncio.run(a.main())
