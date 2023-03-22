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
VERBOSE = False
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
OUTPUT_DIR = os.path.join(str(Path.home() / "Downloads"), "rescore_ui_output")
TODAY = datetime.now().strftime("%Y-%m-%d")
TODAY_DIR = os.path.join(OUTPUT_DIR, TODAY)

now = lambda: datetime.now().strftime("%H-%M-%S")
data_destination_path_raw = lambda: f"{TODAY_DIR}/{now()}_raw-data"
data_destination_path_supervised = lambda: f"{TODAY_DIR}/{now()}_supervised-data"
stats_destination_path = lambda: f"{TODAY_DIR}/{now()}_stats"

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


# Contour prediction parameters
reference_umpp = 0.125
colour_deconv = ColourDeconvolution(
    [
        [0.650, 0.704, 0.286],
        [0.268, 0.570, 0.776],
        # [0, 0, 0]
    ]
)
model = StarDist2D.from_pretrained("2D_versatile_fluo")


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


class ObjectParams():
    def __init__(self, cluster_name: str, cl_i: int, cnt_i: int, area: float, center, disabled: bool):
        self.cluster = cluster_name
        self.cl_i = cl_i
        self.cnt_i = cnt_i
        self.area = area
        self.center = center
        self.disabled = disabled

    def __str__(self) -> str:
        return f"cluster: {self.cluster}{' (disabled)' if self.disabled else ''}; index: {self.cnt_i}; area: {self.area}; center: {self.center}"


class Cluster():
    """ Class for a class of identified objects
    """

    def __init__(self, clustername: str, color, thickness):
        self.name = clustername
        self.color = color
        self.thickness = thickness
        self.checked = True
        self.contours = []
        self.disabled_contours = []

    def sort_contours(self):
        self.contours.sort(key=lambda x: cv.contourArea(x))

    def contour_count(self) -> int:
        return len(self.contours) + len(self.disabled_contours)

    def disable_contour(self, index: int) -> None:
        print("Disable object")
        self.disabled_contours.append(self.contours.pop(index))
        self.disabled_contours.sort(key=lambda x: cv.contourArea(x))

    def enable_contour(self, index: int) -> None:
        print("Enable object")
        self.contours.append(self.disabled_contours.pop(index))
        self.contours.sort(key=lambda x: cv.contourArea(x))

    def to_geojson(self, disableds=False):
        features = []
        for i, contour in enumerate(self.disabled_contours if disableds else self.contours):
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": contour
                },
                "properties": {
                    "index": i,
                    "area": cv.contourArea(contour),
                    "center": centroid_for_contour(contour),
                    "disabled": disableds,
                    "cluster": self.name
                }})
        return {"type": "FeatureCollection", "features": features, "properties": {"cluster": self.name}}


class ImageWindow():
    def __init__(self, name="Interactive UI"):
        self.name = name
        # CREATE CLUSTERS
        self.clusters: list[Cluster] = [
            Cluster("Positive", (107, 76, 254), 2),
            Cluster("Negative", (250, 106, 17), 1),
        ]
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.show_contours = True
        self.refresh_on_next = False
        self.edit = True
        self.stats = None
        self.stats_img = None
        self.selected: ObjectParams = None
        self.data = None

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
        global model
        img = self.og_img.copy()
        cv.imshow(self.name, img)

        concentration_maps = colour_deconv.get_concentration(
            img,
            normalisation="scale"
        )

        for stain_index, cluster in enumerate(self.clusters):
            cluster.contours = list(
                np.rint(
                    predict_contours(
                        model,
                        concentration_maps[..., stain_index],
                        prob_thresh=0.3,
                        nms_thresh=0.1
                    )
                ).astype(int)
            )
            cluster.disabled_contours = []
        self.data = self.extract_dataframe_data()
        return self.clusters

    def toggle_select_mode(self):
        print("SELECT MODE")
        self.edit = False
        self.refresh_on_next = True

    def toggle_edit_mode(self):
        print("EDIT MODE")
        self.edit = True
        self.selected = None
        self.refresh_on_next = True

    def extract_geojson_data(self):
        enabled = [cl.to_geojson() for cl in self.clusters]
        disabled = [cl.to_geojson(True) for cl in self.clusters]
        return [*enabled, *disabled]

    def extract_dataframe_data(self) -> pd.DataFrame:
        features = []
        for feats in self.extract_geojson_data():
            for f in feats["features"]:
                features.append(f)
        cluster = pd.Series([n["properties"]["cluster"] for n in features])
        disabled = pd.Series([n["properties"]["disabled"] for n in features])
        contour_i = pd.Series([n["properties"]["index"] for n in features])
        area = pd.Series([n["properties"]["area"] for n in features])
        center = pd.Series([n["properties"]["center"] for n in features])

        df = pd.DataFrame({
            "cluster": cluster,
            "disabled": disabled,
            "contour_i": contour_i,
            "area": area,
            "center": center
        })
        return df

    def save_data(self, *args):
        data = self.extract_dataframe_data()
        now = datetime.now().strftime("_%H-%M-%S")
        diff = self.data.compare(data)
        if diff.empty:
            data.to_csv(data_destination_path_raw() + ".csv")
        else:
            data.to_csv(data_destination_path_raw() + ".csv")
            data.to_csv(data_destination_path_supervised() + ".csv")
        self.save_stats()
            
    def save_stats(self, *args):
        stats, l = self.extract_stats()
        now = datetime.now().strftime("_%H-%M-%S")
        df = pd.DataFrame({"raw": self.stats["Count"], "supervised": stats["Count"]})

        df.to_csv(stats_destination_path() + ".csv")        

    def update_contour_img(self, segment=True) -> None:
        im = self.og_img.copy()
        if segment:
            self.segmentation()
            self.stats, l = self.extract_stats()

        for cl in self.clusters:
            if not cl.checked:
                continue
            cv.drawContours(im, cl.contours, -1, cl.color, cl.thickness)
            cv.drawContours(im, cl.disabled_contours, -
                            1, (100, 100, 100), cl.thickness)

        if not self.edit:
            if self.selected is not None:
                cl_i, cnt_i = self.selected.cl_i, self.selected.cnt_i
                if self.selected.disabled:
                    cnt = [self.clusters[cl_i].disabled_contours[cnt_i]]
                else:
                    cnt = [self.clusters[cl_i].contours[cnt_i]]
                cv.drawContours(im, cnt, -1, (100, 200, 100), 2, cv.LINE_4)

        self.contour_img = im

        self.refresh_on_next = True
        return im

    def extract_stats(self, *args):
        """Extracts the summary of the clusters' distributions
        """
        enabled = np.sum(
            list(map(lambda cl: len(cl.contours), self.clusters)))
        disabled = np.sum(
            list(map(lambda cl: len(cl.disabled_contours), self.clusters)))
        all = np.sum(
            list(map(lambda cl: cl.contour_count(), self.clusters)))

        indexes1 = list(map(lambda cl: cl.name, self.clusters))

        lengths = list(map(lambda cl: len(cl.contours), self.clusters))

        if not all:
            frame = {"Count": np.zeros(len(lengths)),
                     "Percentage": np.zeros(len(lengths),)}
        else:
            frame = {"Count": pd.Series(lengths, index=indexes1),
                     "Percentage": pd.Series((lengths / enabled) * 100, index=indexes1)}

        count = [enabled, disabled, all]
        percentage = [100 * enabled / all, 100 * disabled / all, int(100)]
        indexes2 = ["enabled", "disabled", "all", ]

        df1 = pd.DataFrame(frame, index=indexes1)
        df2 = pd.DataFrame({"Count": count,
                            "Percentage": percentage}, index=indexes2)
        res = pd.concat(objs=[df1, df2])

        lines = []
        for index, row in res.iterrows():
            lines.append(
                f"{index}: {int(row['Count'])} ({round(row['Percentage'], 2) if bool(row['Count']) else '0'}%)")

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
            cl.contours.pop(params.cnt_i)
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
                obj = find_object_for_point(point, self.clusters, True)

            elif event == 2:
                print("Rescore object")
                obj = find_object_for_point(point, self.clusters, False)
                if obj is not None:
                    self.change_cluster(obj, backwards=False)

            self.refresh_on_next = True
        else:
            if event == 4:
                print("Select object")
                obj = find_object_for_point(point, self.clusters, False)
                self.selected = obj
                obj_stats_img = tb(obj.__str__().split("; "))
                self.stats_img = obj_stats_img
                self.refresh_on_next = True


class BigTing():
    def __init__(self):
        self.context: Context = Context()
        self.socket: Socket = self.context.socket(zmq.PAIR)
        self.socket.bind(ZMQ_SERVERNAME)
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
            self.op = False
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
        while self.op:
            if VERBOSE:
                print(f"zmq iter: {i}")
            i += 1
            message = None
            if self.socket.poll(100, zmq.POLLIN):
                message = self.socket.recv_string()
                self.handle_message(message)
                # print(f"message: \"{message}\"")
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
        while self.op:
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
                    st_im = tb(lines)
                    self.window.stats_img = st_im

                self.change_stats_image()
                self.window.update_contour_img(False)
                self.window.refresh_on_next = False

            key = cv.waitKey(250)

            # Breaks infinite loop if SPACE is pressed OR OpenCV window is closed
            if key == 32 or cv.getWindowProperty(self.window.name, cv.WND_PROP_VISIBLE) < 1:
                self.op = False
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
        cb_vals = [tk.BooleanVar(value=True) for i in clusters]

        def checkbox_update():
            for i, val in enumerate(cb_vals):
                self.window.clusters[i].checked = val.get()
            self.window.refresh_on_next = True

        for i, cluster in enumerate(clusters):
            check_buttons.update({
                cluster.name: tk.Checkbutton(
                    controls_frame, text=cluster.name, variable=cb_vals[i],
                    onvalue=1, offvalue=0, width=10, command=checkbox_update)
            })

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


def find_object_for_point(point: "tuple[int,int]", clusters: "list[Cluster]", disable=False) -> ObjectParams or None:
    hits = []
    for cl_i, cluster in enumerate(clusters):
        if not cluster.checked:
            continue
        for j, contour in enumerate(cluster.contours):
            # If point is on or inside the contour
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:
                hits.append(ObjectParams(cluster.name, cl_i,  j, cv.contourArea(
                    contour), centroid_for_contour(contour), False))

    for cl_i, cluster in enumerate(clusters):
        for j, contour in enumerate(cluster.disabled_contours):
            # If point is on or inside the contour
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:
                hits.append(ObjectParams(cluster.name, cl_i, j, cv.contourArea(
                    contour), centroid_for_contour(contour), True))
    if len(hits) == 1:
        result: ObjectParams = hits[0]
    elif len(hits) == 0:
        return None
    else:
        result: ObjectParams = hits[np.argmin([x.area for x in hits])]

    if disable:
        if result.disabled:
            clusters[result.cl_i].enable_contour(result.cnt_i)
            result.cnt_i = len(clusters[result.cl_i].contours) - 1
        else:
            clusters[result.cl_i].disable_contour(result.cnt_i)
            result.cnt_i = len(clusters[result.cl_i].disabled_contours) - 1
    return result


def centroid_for_contour(contour):
    M = cv.moments(contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return (cx, cy)


def tb(lines: "list[str]"):
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
