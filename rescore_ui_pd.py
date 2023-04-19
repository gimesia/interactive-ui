import asyncio
import csv
from datetime import datetime
import math
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

# Creating Config constants
VERBOSE = True
MAX_WINDOW_WIDTH = 1600
WAIT_TIME = 500

# Dataframe constants
COLUMNS = ["Cluster", "Disabled", "Contour", "Center", "Area"]
MAINFRAME = pd.DataFrame({
    "Cluster": [],
    "Disabled": pd.Series([], dtype=bool),
    "Selected": pd.Series([], dtype=bool),
    "Contour": [],
    "Center": [],
    "Area": []
}, columns=COLUMNS)

# ZMQ message constants
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

# Contour prediction constants
REFERENCE_UMPP = 0.125
COLOUR_DECONVOLUTION = ColourDeconvolution([
    [0.650, 0.704, 0.286],
    [0.268, 0.570, 0.776],
    # [0, 0, 0]
])
MODEL = StarDist2D.from_pretrained("2D_versatile_fluo")


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

    def __init__(self, clustername: str, color: "tuple(int, int, int)", thickness: int):
        self.name = clustername
        self.color = color
        self.thickness = thickness
        self.enabled = True


class ImageWindow():
    def __init__(self, ctrl, name="Interactive UI"):
        self.ctrl: BigTing = ctrl
        # NOTE: circular, because the window is stored
        self.clusters: list[Cluster] = [
            Cluster("Positive", (107, 76, 254), 1),
            Cluster("Negative", (250, 106, 17), 1),
        ]
        self.name = name
        self.raw_df = MAINFRAME.copy()
        self.sample_df = MAINFRAME.copy()
        self.stats: pd.DataFrame = None
        self.stats_img: np.ndarray = None
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.show_contours = True
        self.refresh_on_next = False
        self.edit = True

        # MOCK IMAGE
        self.set_base_image(cv.imread("img/proto_img.tiff"))

        # INITIALIZE OPENCV WINDOW
        cv.namedWindow(self.name, cv.WINDOW_AUTOSIZE)

        # DISPLAY CONTOURS
        self.segment()
        self.update_contours()

        cv.setMouseCallback(self.name, self.mouse_event)

    def get_radius(self):
        return self.ctrl.tk_radius_slider.get()

    def set_base_image(self, img: np.ndarray) -> None:
        """Set new image as reference in the object

        Args:
            img (np.ndarray): 
        """
        self.og_img = img.copy()
        self.contour_img = img.copy()
        self.raw_df = MAINFRAME.copy()
        self.sample_df = MAINFRAME.copy()

    def set_edit_mode(self, val: bool, *args):
        """Radiobutton event handler function, only executes if state is different from 'val' param

        Args:
            val (bool): True if EDIT MODE should be turned on
        """
        if self.edit == val:
            return
        else:
            self.edit = val
            if VERBOSE:
                print(f"EDIT MODE -> {self.edit}")

            # Resetting every row to unselected
            self.sample_df["Selected"] = False
            self.refresh_on_next = True

    def disable_contours(self):
        """Toggles every cluster's visibility off/on
        """
        self.show_contours = not self.show_contours
        self.show_img()

    def show_img(self) -> None:
        """Displays image on window, with or without contours
        """
        if self.show_contours:
            im = self.contour_img.copy()
        else:
            im = self.og_img.copy()

        cv.imshow(self.name, im)  # Display image

        # If image is bigger than the desired value in constants,
        # the window width does not surpass this value
        if self.og_img.shape[1] > MAX_WINDOW_WIDTH:
            self.resize_window()

    def resize_window(self):
        """Downsizes the window to the max window width (keeps aspect ratio)
        """
        sh = self.og_img.shape
        aspect_ratio = sh[0] / sh[1]

        cv.resizeWindow(self.name, MAX_WINDOW_WIDTH,
                        int(MAX_WINDOW_WIDTH * aspect_ratio))

    def segment(self):
        """TODO: Move this step outside this file, it should get only the contours"""
        img = self.og_img.copy()
        self.sample_df = MAINFRAME.copy()

        # Displaying image without contours temporarily
        cv.imshow(self.name, img)

        # Color deconvolution
        concentration_maps = COLOUR_DECONVOLUTION.get_concentration(
            img, normalisation="scale")
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
            contour_series = pd.Series(conts)
            clustername_series = pd.Series(
                np.full_like(contour_series, cluster.name))
            area_series = pd.Series(
                list(map(lambda cnt: cv.contourArea(cnt), conts)))
            centroid_series = pd.Series(
                list(map(lambda cnt: centroid(cnt), conts)))

            # Create df from series
            df = pd.DataFrame({
                "Cluster": clustername_series,
                "Disabled": pd.Series(np.zeros_like(contour_series, dtype=bool)),
                "Selected": pd.Series(np.zeros_like(contour_series, dtype=bool)),
                "Contour": contour_series,
                "Center": centroid_series,
                "Area": area_series
            })

            # Adding rows into object property
            self.raw_df = pd.concat([self.raw_df, df], ignore_index=True)

        self.sample_df = self.raw_df.copy()
        return

    def update_contours(self):
        """Updates the contours on the image based on the 'sample_df' object variable
        """
        im = self.og_img.copy()

        for i in self.sample_df.iterrows():
            clusters = self.clusters
            row = i[1]
            cluster_names = list(map(lambda x: x.name, clusters))

            if not clusters[cluster_names.index(row["Cluster"])].enabled:
                continue

            if row["Selected"]:
                color = (0, 180, 0)
            elif row["Disabled"]:
                color = (100, 100, 100)
            else:
                color = clusters[cluster_names.index(row["Cluster"])].color

            cv.drawContours(im, [row["Contour"]], -1, color, 1, cv.LINE_AA)
        self.contour_img = im
        self.show_img()

    def mouse_event(self, event: int, x: int, y: int, flag: int, *args):
        """Callback function for 

        Args:
            event (int): opencv event identifier
            x (int): x coordinate of click
            y (int): y coordinate of click
            flags (int): opencv event flag identifier
        """
        point = (x, y)
        # EDIT MODE
        if self.edit:
            # DIS-/ENABLE
            if event == cv.EVENT_LBUTTONUP:
                if flag != 8:
                    hit = self.df_polygon_test(point, False)
                    if hit.empty:
                        return

                    # Toggle 'Disabled' value
                    # Get
                    disabled_val = self.sample_df.loc[hit.index].iloc[0]["Disabled"]
                    # Set
                    self.sample_df.loc[hit.index] = not disabled_val

                    self.refresh_on_next = True
                else:  # If ctrl is pressed down, a new entry is created
                    if VERBOSE:
                        print(f"Adding new object")

                    contour = circle_contour(point, self.get_radius())
                    row = {"Cluster": "Negative", "Disabled": False, "Selected": False,
                           "Contour": contour, "Center": (x, y), "Area": None}
                    self.sample_df = self.sample_df.append(
                        row, ignore_index=True)

                    self.refresh_on_next = True

            # RESCORE/DELETE OBJECT
            elif event == cv.EVENT_RBUTTONUP:
                hit = self.df_polygon_test(point, False)

                if hit.empty:
                        return
                
                # Delete if ctrl is pressed down during the click NOTE: bit buggy with the compare after the drop
                """if flag == 8: 
                        if hit.iloc[0]["Area"] == None: # Only delete if it was manually added
                            self.sample_df = self.sample_df.drop(hit.index)
                            self.refresh_on_next = True
                        return
                """
                
                # Get
                clusters = list(map(lambda cl: cl.name, self.clusters))
                cl_name = self.sample_df.loc[hit.index].iloc[0]["Cluster"]
                cl_index = clusters.index(cl_name)
                next_cl_index = (cl_index + 1) % len(clusters)
                # Set
                self.sample_df.loc[hit.index, "Cluster"] = clusters[next_cl_index]
                self.refresh_on_next = True

            else:
                return

        # SELECT MODE
        else:
            # MULTIPLE SELECT WITH LEFTCLICK
            if event == cv.EVENT_LBUTTONUP:
                hit = self.df_polygon_test(point, False)

                if hit.empty:
                    return

                self.sample_df["Selected"] = False
                self.sample_df.loc[hit.index, "Selected"] = True

                self.refresh_on_next = True
            # SINGLE SELECT WITH RIGHTCLICK
            # TODO give other functionality (e.g. multiselect)
            elif event == cv.EVENT_RBUTTONUP:
                hit = self.df_polygon_test(point, False)

                if hit.empty:
                    return

                self.sample_df["Selected"] = False
                self.sample_df.loc[hit.index, "Selected"] = True

                self.refresh_on_next = True
            else:
                return

    def df_polygon_test(self, point: "tuple(int, int)", multiple=True):
        """Returns the rows that contain the point given as parameter

        Args:
            point (tuple): (x, y) coordinates of a point
            multiple (bool, optional): return all rows that contain the point. Defaults to True.

        Returns:
            pd.DataFrame: dataframe of hit(s)
        """
        condition = self.sample_df["Contour"].apply(
            lambda x: cv.pointPolygonTest(x, point, False) >= 0
        )
        hits = self.sample_df.loc[condition]

        if multiple:
            return hits

        mn = hits["Area"].min()

        # If there is 'Area' for any of the hits, the smallest is returned
        if not pd.isna(mn):
            condition2 = hits["Area"].apply(lambda x: x == mn)
            mn_hit = hits[condition2]
            return mn_hit

        # If the minimum isnan (doesn't have Area), that means the obj was added manually
        else:
            if len(hits) < 2:
                return hits
            else:
                hits_ = hits.copy()
                # Separating coordinates
                hits_["X"], hits_["Y"] = zip(*hits_["Center"])
                # Calculating distance (Euclidean)
                hits_["Euclidean_distance"] = np.sqrt(
                    (hits_["X"] - point[0]) ** 2 + (hits_["Y"] - point[1]) ** 2)
                # Selecting the row with the shortest distance
                mn_dst = hits_["Euclidean_distance"].min()
                condition3 = hits_["Euclidean_distance"].apply(
                    lambda x: x == mn_dst)
                hits_ = hits_[condition3]
                # Dropping helper columns
                hits_ = hits_.drop(columns=["X", "Y", "Euclidean_distance"])

                return hits_

    def extract_data(self):
        """Extracts data into destination files found in the end of the module
        """
        if self.extract_changelog().empty:
            if VERBOSE:
                print(f"Extracting unmodified data")
            self.raw_df.to_csv(data_destination_path_raw() + ".csv")

        else:
            if VERBOSE:
                print(f"Extracting modified data")
            self.raw_df.to_csv(data_destination_path_raw() + ".csv")
            self.sample_df.to_csv(data_destination_path_supervised() + ".csv")

    def extract_stats(self):
        """Extracts the summarized stats and saves it into destination file
        """
        if self.extract_changelog().empty:
            stats_df = self.stats_for_df(self.sample_df)

        else:
            res1 = self.stats_for_df(self.raw_df)["count"]
            res2 = self.stats_for_df(self.sample_df)["count"]

            stats_df = pd.DataFrame(
                {"Raw": res1, "Supervised": res2}, index=res1.index)

            if VERBOSE:
                print("Stats:\n")
                print(stats_df)
            stats_df.to_csv(stats_destination_path() + ".csv")
        return stats_df

    def stats_for_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extracts the statistics from an input df

        Args:
            df (pd.DataFrame): input dataframe

        Returns:
            pd.DataFrame: dataframe of the statistics
        """

        clusters = df["Cluster"].unique()
        counts = []

        for i in clusters:
            counts.append(
                len(df[(df["Cluster"] == i) & (df["Disabled"] == False)])
            )

        disabled = len(df[(df["Disabled"] == True)])
        enabled = len(df[(df["Disabled"] == False)])
        all = len(df)

        counts = [*counts, disabled, enabled, all]
        indexes = [cl for cl in clusters]
        indexes = [*indexes, "Disabled", "Enabled", "All"]

        result = (pd.DataFrame({}, index=indexes))
        result["count"] = counts
        return result

    def extract_changelog(self):
        """Extracts the changes made during the supervision
        """
        # calculate difference
        len_diff = len(self.sample_df) - len(self.raw_df)

        if len_diff != 0:  # if sample df has more rows then the raw
            filled_raw = self.raw_df.copy()
            self.sample_df.columns = self.raw_df.columns

            for i in range(len_diff):  # Fill with empty rows
                filled_raw = filled_raw.append(pd.Series(
                    [None]*len(filled_raw.columns), index=filled_raw.columns), ignore_index=True)

            print(f"raw:\n{len(filled_raw)}")
            print(f"sample:\n{len(self.sample_df)}")
            diff = filled_raw.compare(self.sample_df)

        else:
            diff = self.raw_df.compare(self.sample_df)

        return diff

    def extract_selected(self) -> pd.DataFrame:
        """Extracts the selected object

        Returns:
            pd.DataFrame: dataframe 
        """
        res = self.sample_df[self.sample_df["Selected"] == True]
        res = res.drop(["Contour", "Selected"], axis=1)
        res = res.transpose()

        return res


class BigTing():
    def __init__(self):
        self.context: Context = Context()
        self.socket: Socket = self.context.socket(zmq.PAIR)
        self.window = ImageWindow(self)
        self.is_open = True
        self.tk: tk.Tk = None
        self.tk_radius_slider = None
        self.tk_text = None  # Displayed text widget

        self.socket.bind(ZMQ_SERVERNAME)

    def change_stats(self):
        """Changes the displayed stats based on the mode
        """
        if self.window.edit:
            a = self.window.extract_stats().to_string()
        else:
            a = self.window.extract_selected()
            if a.empty:
                a = self.window.extract_stats()
            a = a.to_string()

        self.tk_text.config(state=tk.NORMAL)
        self.tk_text.delete('1.0', 'end')
        self.tk_text.insert(tk.END, a)
        self.tk_text.config(state=tk.DISABLED)
        self.tk_text.pack()
        if VERBOSE:
            print("Changing displayed statistics")

    def handle_message(self, msg: str):
        """Chooses appropriate response based on incoming message

        Args:
            msg (str): incoming message
        """
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
            self.window.segment()
            self.window.show_img()
        except:
            self.confirm_req_failed()
        self.window.refresh_on_next = True

    def send_contours(self):  # Sending the array of the contours
        print("Sending contours")
        cl = [cl.name for cl in self.window.clusters]
        df = self.window.sample_df

        contours = [df[df["Cluster"] == c]["Contour"] for c in cl]
        self.socket.send_pyobj(contours)

        if VERBOSE:
            print(contours)

    async def coroutine_zmq(self):
        """ZeroMQ communication async coroutine"""
        print("ZMQ Coroutine is running")
        i = 0
        while self.is_open:
            if VERBOSE:
                print(f"zmq iter: {i}")
            i += 1
            message = None
            if self.socket.poll(WAIT_TIME, zmq.POLLIN):
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
                self.window.update_contours()

                self.change_stats()

                self.window.refresh_on_next = False

            key = cv.waitKey(WAIT_TIME)

            # Breaks infinite loop if SPACE is pressed OR OpenCV window is closed
            if key == 32 or cv.getWindowProperty(self.window.name, cv.WND_PROP_VISIBLE) < 1:
                self.is_open = False
                self.tk.quit()
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

        text_frame = tk.Frame(root)
        text_frame.pack(side="left", fill="both", padx=12, pady=12)

        text = tk.Text(text_frame, font=("Helvetica", 16), width=20, height=8)

        text.config(state=tk.NORMAL)
        text.insert(tk.END, self.window.extract_stats().to_string())
        text.config(state=tk.DISABLED)
        self.tk_text = text
        self.tk_text.pack()

        controls_frame = tk.Frame(root)
        controls_frame.pack(fill="both", expand=True, padx=(0, 5), pady=5)

        rad_btn_frame = tk.Frame(controls_frame)
        rad_btn_val = tk.BooleanVar()
        rad_btn_frame.pack(expand=True)

        def toggle_select():
            self.window.set_edit_mode(False)
            self.tk_radius_slider.pack_forget()

        def toggle_edit():
            self.window.set_edit_mode(True)
            self.tk_radius_slider.pack()

        R1 = tk.Radiobutton(rad_btn_frame, text="SELECT",
                            variable=rad_btn_val, value=True, command=toggle_select)
        R2 = tk.Radiobutton(rad_btn_frame, text="EDIT",
                            variable=rad_btn_val, value=False, command=toggle_edit)
        R1.pack(anchor=tk.W, side=tk.LEFT)
        R2.pack(anchor=tk.W, side=tk.LEFT)

        def update_checkboxes():
            # Set each cluster.enabled value t
            for i, key in enumerate(checkbox_vals):
                val = checkbox_vals[key].get()
                self.window.clusters[i].enabled = val
            self.window.refresh_on_next = True

        checkbox_vals = {i.name: tk.BooleanVar(
            value=True) for i in self.window.clusters}
        checkboxes = {
            i: tk.Checkbutton(
                controls_frame, text=i, variable=checkbox_vals[i], onvalue=True, offvalue=False, command=update_checkboxes
            ) for i in checkbox_vals.keys()
        }

        for cb in checkboxes.values():
            cb.pack()

        btn_frame = tk.Frame(controls_frame)
        btn_frame.pack(fill="both", pady=15)

        button1 = tk.Button(btn_frame, text="Save STATS",
                            command=self.window.extract_stats)
        button2 = tk.Button(btn_frame, text="Save DATA",
                            command=self.window.extract_data)
        button3 = tk.Button(btn_frame, text="Dis-/Enable Contours",
                            command=self.window.disable_contours)
        button4 = tk.Button(btn_frame, text="Changelog",
                            command=self.window.extract_changelog)
        button1.grid(row=0, column=0, padx=5, pady=2)
        button2.grid(row=1, column=0, padx=5, pady=2)
        button3.grid(row=0, column=1, padx=5, pady=2)
        button4.grid(row=1, column=1, padx=5, pady=2)

        self.tk_radius_slider = tk.Scale(
            controls_frame, from_=5, to=15, length=200, orient=tk.HORIZONTAL, label="Radius for new object")
        self.tk_radius_slider.pack(fill="both", expand=True)

        def on_closing():
            self.is_open = False
            self.tk.quit()
        root.protocol("WM_DELETE_WINDOW", on_closing)

        root.mainloop()

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


def circle_contour(center: "tuple(int, int)", radius: int):
    """Calculates the opencv contour of a circle

    Args:
        center (tuple): origin of the circle
        radius (int): length of the radius

    Returns:
        np.ndarray: array of the polygons points 
    """
    points = []
    for i in range(0, 360):
        angle = i * math.pi / 180
        x = int(center[0] + radius * math.cos(angle))
        y = int(center[1] + radius * math.sin(angle))
        points.append([x, y])

    # Convert points to NumPy array
    points = np.array(points)

    # Reshape to fit cv2.drawContours input format
    points = points.reshape((-1, 1, 2))
    return points


# Output Constants
OUTPUT_DIR = os.path.join(str(Path.home() / "Downloads"), "rescore_ui_output")
TODAY = datetime.now().strftime("%Y-%m-%d")
TODAY_DIR = os.path.join(OUTPUT_DIR, TODAY)


# Dynamic functions for saving results
def now(): return datetime.now().strftime("%H-%M")
def data_destination_path_raw(): return f"{TODAY_DIR}/{now()}_raw-data"
def data_destination_path_supervised(
): return f"{TODAY_DIR}/{now()}_supervised-data"
def stats_destination_path(): return f"{TODAY_DIR}/{now()}_stats"


# Creating output files
def create_dist_lib():
    # TODO: further isolation of saved data of different quants
    try:
        os.listdir(OUTPUT_DIR)
    except:
        os.mkdir(OUTPUT_DIR)
    try:
        os.listdir(TODAY_DIR)
    except:
        os.mkdir(TODAY_DIR)


if __name__ == "__main__":
    create_dist_lib()
    bt = BigTing()
    bt.run()  # run the asyncio program
