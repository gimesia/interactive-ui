import json
import tkinter
import cv2 as cv
import zmq
import sys
from tkinter import ttk
from PIL import Image, ImageTk


class Cluster():
    def __init__(self, clustername, color):
        self.cluster_name = clustername
        self.color = color
        self.checked = False
        self.contours = []
        self.on_checked_changed = None

    def set_checked_state(self):
        self.checked = not self.checked
        if self.on_checked_changed is not None:
            self.on_checked_changed(self)

    def subscribe_on_checked_changed_event(self, function):
        self.on_checked_changed = function


class Model():
    def __init__(self):
        self.base_image = None
        self.modified_image = None
        self.block_size = 101
        self.c_value = 21
        self.clusters = [Cluster('cl1', (0, 0, 255)), Cluster(
            'cl2', (255, 255, 0)), Cluster('cl3', (255, 136, 0)), Cluster('cl4', (255, 0, 0))]
        self.block_size_on_changed = []
        self.c_value_on_changed = []

    def subscribe_block_size_changed_event(self, function):
        self.block_size_on_changed.append(function)

    def subscribe_c_value_changed_event(self, function):
        self.c_value_on_changed.append(function)

    def set_base_image(self, image):
        self.base_image = image.copy()
        self.modified_image = image.copy()

    def set_block_size(self, value):
        self.block_size = value
        for function in self.block_size_on_changed:
            function()

    def set_c_value(self, value):
        self.c_value = value
        for function in self.c_value_on_changed:
            function()

    def set_values(self, blocksize, cvalue):
        self.block_size = blocksize
        self.c_value = cvalue


class View(ttk.Frame):
    checkButtons = {}

    def __init__(self, parent, model):
        super().__init__(parent)
        self.model = model
        self.pack(side='right', fill='y', padx=5)

        checkButtonFrame = tkinter.Frame(self)
        checkButtonFrame.pack(side='right', fill='y')
        tkinter.Label(checkButtonFrame, text='Clusters').pack()
        for value in model.clusters:
            self.checkButtons[value.cluster_name] = tkinter.Checkbutton(
                checkButtonFrame, text=value.cluster_name, command=lambda x=value.cluster_name: self.checked_changed(x))
            self.checkButtons[value.cluster_name].pack()
            self.checkButtons[value.cluster_name].deselect()

        blockSizeFrame = tkinter.Frame(checkButtonFrame)
        blockSizeFrame.pack()
        blockSizeLabel = ttk.Label(blockSizeFrame, text='Block size')
        blockSizeLabel.pack(anchor=tkinter.S, side='left')
        self.block_size = tkinter.DoubleVar()
        self.blockSizeScale = tkinter.Scale(blockSizeFrame, from_=3, to=105, length=150,
                                            variable=self.block_size, orient=tkinter.HORIZONTAL, command=self.block_size_changed)
        self.blockSizeScale.pack(anchor='w', side='right')
        self.block_size.set(self.model.block_size)
        CValueFrame = tkinter.Frame(checkButtonFrame)
        CValueFrame.pack()
        CValueLabel = ttk.Label(CValueFrame, text='C value')
        CValueLabel.pack(anchor=tkinter.S, side='left')
        self.c_value = tkinter.DoubleVar()
        self.CValueLabelScale = tkinter.Scale(CValueFrame, from_=1, to=100, length=150,
                                              variable=self.c_value, orient=tkinter.HORIZONTAL, command=self.c_value_changed)
        self.CValueLabelScale.pack(anchor='w', side='right')
        self.c_value.set(self.model.c_value)
        self.done_button = tkinter.Button(checkButtonFrame, text='Done')
        self.done_button.pack(anchor='c', pady=(10, 0))

        self.imageFrame = tkinter.Frame(self)
        self.imageFrame.pack(side='left', fill='y')
        self.imagelabel = None

    def block_size_changed(self, value):
        self.model.set_block_size(int(value))

    def c_value_changed(self, value):
        self.model.set_c_value(int(value))

    def checked_changed(self, name):
        cluster = next(
            (x for x in self.model.clusters if x.cluster_name == name), None)
        cluster.set_checked_state()

    def refresh_photo(self):
        im = Image.fromarray(self.model.modified_image)
        # resized_image = im.resize((1600,1200))
        self.photo = ImageTk.PhotoImage(image=im)
        if self.imagelabel is not None:
            self.imagelabel.destroy()
        self.imagelabel = tkinter.Label(self.imageFrame, image=self.photo)
        self.imagelabel.pack()


class Controller():
    selected_clusters = []

    def __init__(self, model, view, socket, ispreview):
        self.model = model
        self.view = view
        self.socket = socket
        self.is_preview = ispreview

        if self.is_preview:
            for i in self.model.clusters:
                i.subscribe_on_checked_changed_event(self.checked)

            self.model.subscribe_block_size_changed_event(
                self.apply_adaptive_threshold)
            self.model.subscribe_c_value_changed_event(
                self.apply_adaptive_threshold)

            self.view.done_button.bind(
                "<ButtonRelease-1>", lambda a: self.send_info())

    def apply_adaptive_threshold(self):
        if self.model.base_image is not None:
            block_size = self.model.block_size if self.model.block_size % 2 == 1 else self.model.block_size+1
            contours = self.calculate_adaptive_threshold(
                self.model.base_image, block_size, self.model.c_value)
            for mc in self.model.clusters:
                mc.contours.clear()
            for contour in contours:
                if len(contour) >= 3:
                    self.model.clusters[len(contour) %
                                        4].contours.append(contour)

            if self.view is not None:
                self.remove_contours()

    def calculate_adaptive_threshold(self, image, blocksize, cvalue):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        thr = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, blocksize, cvalue)
        return cv.findContours(thr, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

    def apply_contours(self, cluster, refresh=True):
        if len(cluster.contours) > 0:
            cv.drawContours(self.model.modified_image,
                            cluster.contours, -1, cluster.color, 3)
            if refresh:
                self.view.refresh_photo()

    def remove_contours(self):
        if self.model.base_image is not None:
            self.model.modified_image = self.model.base_image.copy()
            for i in self.selected_clusters:
                self.apply_contours(i, False)

            self.view.refresh_photo()

    def checked(self, cluster):
        if cluster.checked:
            self.apply_contours(cluster)
            self.selected_clusters.append(cluster)
        else:
            self.selected_clusters.remove(cluster)
            self.remove_contours()

    def send_info(self):
        if self.model.base_image is not None:
            for cluster in self.model.clusters:
                if self.socket.poll(1000, zmq.POLLIN):
                    req = self.socket.recv_string()
                self.socket.send_pyobj(cluster.contours)

            message = self.socket.recv_string()
            if message != "DONE":
                info = {
                    'block_size': self.model.block_size,
                    'c_value': self.model.c_value
                }
                self.socket.send_string(json.dumps(info))
            else:
                self.socket.send_string("OK")


class App(tkinter.Tk):
    PING = 'PING'
    THRESHOLD_INFO = 'THRESHOLD_INFO'
    INP_IMAGE = 'INP_IMAGE'
    QUERY_CONTOUR = 'QUERY_CONTOUR'
    DONE = 'DONE'
    ALIVE = 'ALIVE'
    EXIT = 'EXIT'
    UNKNOWN = 'UNKNOWN'

    def __init__(self):
        if len(sys.argv) > 1:
            self.is_preview = True if sys.argv[1] == 'True' else False
        else:
            self.is_preview = True

        if self.is_preview:
            super().__init__()

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind('tcp://*:5555')
        self.model = Model()

        if len(sys.argv) > 2:
            data = sys.argv[2]
            if data != '':
                info = json.loads(data)
                self.model.set_values(info['block_size'], info['c_value'])

        if self.is_preview:
            self.attributes('-topmost', True)
            self.geometry("1920x1017")
            self.title("Interactive ui")
            self.view = View(self, self.model)
        else:
            self.view = None

        self.controller = Controller(
            self.model, self.view, self.socket, self.is_preview)

    def listen(self):
        if self.socket.poll(100, zmq.POLLIN):
            message = self.socket.recv_string()
            if message == self.PING:
                self.pong()
            elif message == self.THRESHOLD_INFO:
                self.receive_threshold_info()
            elif message == self.INP_IMAGE:
                self.receive_image()
            elif message == self.QUERY_CONTOUR:
                if not self.is_preview:
                    self.controller.send_info()
            else:
                self.unknown_message()

    def receive_image(self):
        self.socket.send_string(self.DONE)
        message = self.socket.recv_pyobj()
        self.model.set_base_image(message)
        self.controller.apply_adaptive_threshold()
        if self.view is not None:
            self.view.refresh_photo()
        self.socket.send_string(self.DONE)

    def receive_threshold_info(self):
        self.socket.send_string(self.DONE)
        message = self.socket.recv_string()
        data = json.loads(message)
        self.model.set_values(data['block_size'], data['c_value'])
        self.socket.send_string(self.DONE)

    def pong(self):
        self.socket.send_string(self.ALIVE)

    def unknown_message(self):
        self.socket.send_string(self.UNKNOWN)

    def dispose(self):
        if self.socket.poll(100):
            self.socket.recv_string()
        self.socket.send_string(self.EXIT)
        self.destroy()


is_open = True


def close_window():
    global is_open
    is_open = False


if __name__ == '__main__':
    app = App()
    if app.is_preview:
        app.protocol("WM_DELETE_WINDOW", close_window)
    while is_open:
        if app.is_preview:
            app.update_idletasks()
            app.update()
        app.listen()

    app.dispose()
