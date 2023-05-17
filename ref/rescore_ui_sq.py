info = {"title": "Interactive UI", "requirements": ["pyzmq", "pandas"]}
# This example presents how zmq and subprocess can be used to create an external process, and communicate with it.
# External process makes the calculation, and displays the result. If the user done with the settings of the threshold info,
# the calculated contours and threshold parameters can be transferred back to ScriptQuant.
# External process get started automatically by this script.
# External process code can be found in ScriptQuant.Examples/interactive_ui.py.
from pathlib import Path
from ast import Return
import quantification as qc
import numpy as np
import zmq
import datetime
import subprocess
import os
from datetime import date

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
with open(f"{OUTPUT_DIR}\\log.txt", "w") as f:
    f.write("Created log file!\n")


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
    out.clusters.add("Positive", "#FF0000").add("Negative", "#0000FF")
    out.processing.tile_size = 1024
    out.processing.tile_border_size = 128
    out.processing.zoom = 2.5
    isPreview = inp.environment.is_preview_segmentation
    pythonPath = inp.environment.python_path
    externalprocesspath = inp.environment.app_path + \
        "\\ScriptQuant.Examples\\Usecases\\interactive_ui_cv.py"
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
        communicate("PING")
    except ExternalProcessNotRespondingException:
        print("External process is not responding. ScriptQuant tries to start the process.")
    finally:
        start_process(threshold_info)


def process_tile(inp: qc.ProcessInput, out: qc.ProcessOutput):
    try:
        # Send the image to external process, which is making some calculations, and sends back the contours.
        communicate("INP_IMAGE")
        communicate(inp.image)
        contours = communicate(
            "QUERY_CONTOUR", returntype=RETURN_TYPE_PYOBJ, timeout=720000)
        for i, cl in enumerate(contours):
            for polygon in cl:
                points = []
                for j, p in enumerate(polygon):
                    if j == len(polygon) - 1:
                        break
                    points.append((p[0], p[1]))
                out.results.add_polygon(
                    i, points, custom_data=datetime.datetime.now().strftime("%c"))

        communicate("DONE")

    except ExternalProcessNotRespondingException as e:
        print("External process is not responding.")
    except Exception as e:
        print("External process has exited.")
        print(e)

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
