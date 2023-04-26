info = {"title": "Interactive UI", "requirements": ["pyzmq", "pandas"]}
# This example presents how zmq and subprocess can be used to create an external process, and communicate with it.
# External process makes the calculation, and displays the result. If the user done with the settings of the threshold info,
# the calculated contours and threshold parameters can be transferred back to ScriptQuant.
# External process get started automatically by this script.
# External process code can be found in ScriptQuant.Examples/interactive_ui.py.
from pathlib import Path
from ast import Return
import time
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

class Cluster():
    """Class for a class of identified objects"""
    def __init__(self, clustername: str, color: "tuple(int, int, int)", thickness: int=1):
        self.name = clustername
        self.color = color
        self.thickness = thickness
        self.enabled = True

REQUEST_RETRIES = 1
REQUEST_TIMEOUT = 1000
SERVER_ENDPOINT = "tcp://localhost:5560"
RETURN_TYPE_PYOBJ = 1
RETURN_TYPE_STRING = 2
RETURN_TYPE_UNKNOWN = 3
INP_IMG = "INP_IMAGE"
SEND_CNTS = "SEND_CONTOURS"

context = None
socket = None
externalprocesspath = None
external_process = None
logfile = None
isPreview = False
pythonPath = ""

# Output Constants
OUTPUT_DIR = os.path.join(str(Path.home() / "Downloads"), "rescore_ui_output")
TODAY = datetime.datetime.now().strftime("%Y-%m-%d") # datetime.now().strftime("%Y-%m-%d")
TODAY_DIR = os.path.join(OUTPUT_DIR, TODAY)

LOGFILE_PATH = f"{OUTPUT_DIR}\\log.txt"

CLUSTERS = [Cluster("pos", (0, 0, 255)), Cluster("neg", (0, 255, 0))]

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



# Start the external process with argument.
def start_process(inp: qc.InitializeInput, contours: np.ndarray):
    global logfile, external_process, externalprocesspath, pythonPath, context, socket
    
    create_dist_lib()
    
    # A logfile created, and passed to the subprocess, so it can log any error into this file.
    logfile = open(LOGFILE_PATH, "a")
    logfile.write(
        "-----------------------NEW PROCESS-----------------------\n")
    logfile.write(f"{datetime.datetime.now()}:\n")
    logfile.flush()
    

    socket.setsockopt(zmq.LINGER, 100)
    socket.connect(SERVER_ENDPOINT)

    external_process_start_command = [pythonPath + "\\python.exe", externalprocesspath, str(isPreview)]
    external_process = subprocess.Popen(external_process_start_command, stderr=logfile, creationflags=subprocess.CREATE_NO_WINDOW)

    time.sleep(1)

    try:
        communicate("PING")
    except ExternalProcessNotRespondingException:
        print("External process is not responding. ScriptQuant tries to start the process.")
    finally:
        communicate(INP_IMG)
        communicate(inp.image)
        communicate(SEND_CNTS)
        communicate(contours)
        
def initialize(inp: qc.InitializeInput, out: qc.InitializeOutput):
    global context, socket, isPreview, pythonPath, externalprocesspath
    
    isPreview = inp.environment.is_preview_segmentation
    pythonPath = inp.environment.python_path
    externalprocesspath = OUTPUT_DIR + "\\rescore_ui_pd.py"

    context = zmq.Context()
    socket = context.socket(zmq.PAIR)

    for i in CLUSTERS:
        out.clusters.add(
            i.name,
            '#{:02x}{:02x}{:02x}'.format(i.color[0], i.color[1], i.color[2])
        ) 

    out.processing.tile_size = 1024
    out.processing.tile_border_size = 128
    out.processing.zoom = 2.5
    isPreview = inp.environment.is_preview_segmentation



def process_tile(inp: qc.ProcessInput, out: qc.ProcessOutput):
    concentration_maps = COLOUR_DECONVOLUTION.get_concentration(inp.image, normalisation="scale")
    contours = []
    for stain_index, cluster in enumerate(CLUSTERS):
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
        
        contours.append(conts)
        
        for i, cnt in enumerate(conts):
            points = []
            for point in cnt:
                points.append((point[0], point[1]))
            points.pop()
        
            out.results.add_polygon(
            stain_index, points, custom_data=datetime.datetime.now().strftime("%c"))
    
    if isPreview:
        start_process(inp, contours)


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
 
