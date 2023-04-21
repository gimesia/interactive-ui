info = {"title": "Interactive UI", "requirements": ["pyzmq", "pandas"]}
# This example presents how zmq and subprocess can be used to create an external process, and communicate with it.
# External process makes the calculation, and displays the result. If the user done with the settings of the threshold info,
# the calculated contours and threshold parameters can be transferred back to ScriptQuant.
# External process get started automatically by this script.
# External process code can be found in ScriptQuant.Examples/interactive_ui.py.
from pathlib import Path
from ast import Return
from turtle import pd
import quantification as qc
import numpy as np
import zmq
import datetime
import subprocess
import os
from datetime import date
from stardist.models import StarDist2D
from stardist.nms import non_maximum_suppression
from stardist.geometry import dist_to_coord
from colour_deconvolution import ColourDeconvolution


REQUEST_RETRIES = 1
REQUEST_TIMEOUT = 1000
SERVER_ENDPOINT = "tcp://localhost:5560"
RETURN_TYPE_PYOBJ = 1
RETURN_TYPE_STRING = 2
RETURN_TYPE_UNKNOWN = 3

context = None
socket = None
externalprocesspath = None
external_process = None
logfile = None
isPreview = False
pythonPath = ""


# Output Constants
OUTPUT_DIR = os.path.join(str(Path.home() / "Downloads"), "rescore_ui_output")
TODAY = datetime.datetime.now().strftime("%Y-%m-%d")#datetime.now().strftime("%Y-%m-%d")
TODAY_DIR = os.path.join(OUTPUT_DIR, TODAY)



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

create_dist_lib()
logfilepath = f"{OUTPUT_DIR}\\log.txt"
with open(f"{OUTPUT_DIR}\\log.txt", "a") as f:
    pass

class Cluster():
    """Class for a class of identified objects"""
    def __init__(self, clustername: str, color: "tuple(int, int, int)", thickness: int):
        self.name = clustername
        self.color = color
        self.thickness = thickness
        self.enabled = True

clusters = [Cluster("pos", (0, 0, 255)), Cluster("neg", (0, 0, 255))]

# Start the external process with argument.
def start_process(thresholdinfo):
    global logfile, external_process
    # A logfile created, and passed to the subprocess, so it can log any error into this file.
    logfile = open(logfilepath, "a")
    logfile.write(
        "-----------------------NEW PROCESS-----------------------\n")
    logfile.write(f"{datetime.datetime.now()}:\n")
    logfile.flush()

    # external_process_start_command = [
    #     pythonPath + "\\python.exe", externalprocesspath, str(isPreview), thresholdinfo]
    # external_process = subprocess.Popen(
    #     external_process_start_command, stderr=logfile, creationflags=subprocess.CREATE_NO_WINDOW)


def initialize(inp: qc.InitializeInput, out: qc.InitializeOutput):
    global context, socket, isPreview, pythonPath, externalprocesspath
    # .add("cl3", "#FFFF00").add("cl4", "#FF8800")
    # out.clusters.add("Positive", "#FF0000").add("Negative", "#0000FF")
    for i in clusters:
        out.clusters.add(i.name, i.color) 
    out.processing.tile_size = 1024
    out.processing.tile_border_size = 128
    out.processing.zoom = 2.5
    isPreview = inp.environment.is_preview_segmentation
   
    # pythonPath = inp.environment.python_path
    # externalprocesspath = inp.environment.app_path + \
    #     "\\ScriptQuant.Examples\\Usecases\\interactive_ui_cv.py"
    # Connect to the socket.
  
    context = zmq.Context()
    socket = context.socket(zmq.PAIR)
    socket.setsockopt(zmq.LINGER, 100)
    socket.connect(SERVER_ENDPOINT)

    # Check if any data saved into the scenario.
    threshold_info = ""
    if inp.saved_data != "":
        threshold_info = inp.saved_data

    # Check if external process is alive, if not, then start it.
    try:
        # communicate("PING")
        print(out.clusters)
        start_process(threshold_info)
    except ExternalProcessNotRespondingException:
        print("External process is not responding. ScriptQuant tries to start the process.")
    finally:
        start_process(threshold_info)


def process_tile(inp: qc.ProcessInput, out: qc.ProcessOutput):
    concentration_maps = COLOUR_DECONVOLUTION.get_concentration(inp.image, normalisation="scale")
    for stain_index, cluster in enumerate(out.clusters):
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
    for i, cl in enumerate(conts):
        for polygon in cl:
            points = []
            for j, p in enumerate(polygon):
                if j == len(polygon) - 1:
                    break
                points.append((p[0], p[1]))
            out.results.add_polygon(
                i, points, custom_data=datetime.datetime.now().strftime("%c"))

# This function only be used for close, delete all reserved resources.
# Called when script/scenario/quantcenter closed, or in processing after end_tiling function finished.


def deinitialize():
    if logfile is not None:
        logfile.close()
    if external_process is not None:
        external_process.terminate()
    socket.close()
    context.term()

# Communicate with the external process via zmq.
# If the communication fails due to the external process is not responding, this function closes the connection and empties the message queue, and
# then reconnect. This is neccessary, because if a message stuck in the queue, this process cannot send any message until the server not responding.
# This can cause crashes in ScriptQuant.
# The request parameter is what we want to send to the server. It could be also a string or a pyobject.
# With the REQUEST_RETRIES constant variable we can set how many times the zmq should try to communicate with the external process.
# With the REQUEST_TIMEOUT constant variable we can set the timeout how long the zmq should wait for a response.
# If the communicate fails, then throws an ExternalProcessNotResponding exception.


def communicate(request, returntype=RETURN_TYPE_STRING, timeout=REQUEST_TIMEOUT):
    global socket
    retries_left = REQUEST_RETRIES
    reply = None
    if type(request) is str:
        print(f"Request: {request}")

    while retries_left != 0:
        if isinstance(request, str):
            socket.send_string(request)
        else:
            socket.send_pyobj(request)

        if socket.poll(timeout, zmq.POLLIN):
            if returntype == RETURN_TYPE_STRING:
                reply = socket.recv_string()
                print(f"Reply: {reply}")
            elif returntype == RETURN_TYPE_PYOBJ:
                reply = socket.recv_pyobj()
            elif returntype == RETURN_TYPE_UNKNOWN:
                reply = socket.recv()
                print(f"Reply: {reply}")
            return reply

        retries_left -= 1
        socket.setsockopt(zmq.LINGER, 100)
        socket.close()
        socket = context.socket(zmq.PAIR)
        socket.connect(SERVER_ENDPOINT)

        if retries_left == 0:
            raise ExternalProcessNotRespondingException()


class ExternalProcessNotRespondingException(Exception):
    pass

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

