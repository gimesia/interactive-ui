import numpy as np
import cv2 as cv


class Cluster():
    """ Class for a class of identified objects
    """

    def __init__(self, clustername, color):
        self.cluster_name = clustername
        self.color = color
        self.checked = False
        self.contours = []
        self.on_checked_changed = None

    def subscribe_on_checked_changed_event(self, function):
        self.on_checked_changed = function

    def set_checked_state(self):
        self.checked = not self.checked
        if self.on_checked_changed is not None:
            self.on_checked_changed(self)


class ImageWindow():
    def __init__(self):
        self.clusters: list[Cluster] = []
        self.og_img: np.ndarray = None
        self.modified_img: np.ndarray = None
        self.block_size = 101
        self.c_val = 21
        self.block_size_on_changed = []
        self.c_value_on_changed = []
        self.window = None

    def open_window(self):
        self.window = 'Interactive UI'
        cv.namedWindow(self.window)
        cv.imshow(self.window, self.modified_img)
        # Sliders
        max_thresh = 255
        thresh = 100  # initial threshold
        cv.createTrackbar('Block size:', self.window,
                          self.block_size, max_thresh, self.set_block_size)
        cv.createTrackbar('C value:', self.window,
                          self.c_val, max_thresh, self.set_c_value)

        cv.createButton('1', c_val_callback, None, cv.QT_CHECKBOX, 1),
        cv.createButton('2', c_val_callback, None, cv.QT_CHECKBOX, 2),
        cv.createButton('3', c_val_callback, None, cv.QT_CHECKBOX, 3),

        cv.waitKey(0)
        cv.destroyAllWindows()

    def subscribe_block_size_changed_event(self, func):
        self.block_size_on_changed.append(func)

    def subscribe_c_value_changed_event(self, function):
        self.c_value_on_changed.append(function)

    def set_base_image(self, img: np.ndarray):
        self.og_img = img.copy()
        self.modified_img = img.copy()

    def set_block_size(self, val):
        print(f"Block size set to: {val}")
        self.block_size = val
        for function in self.block_size_on_changed:
            function()

    def set_c_value(self, val):
        print(f"C value set to: {val}")
        self.c_val = val
        for function in self.c_value_on_changed:
            function()

    def calculate_adaptive_th(self):
        gray_img = cv.cvtColor(self.og_img, cv.COLOR_BGR2GRAY)
        th = cv.adaptiveThreshold(
            gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, self.blocksize, self.c_val)
        return cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]


def block_size_callback(event):
    """Callback function for the block size value slider

    Args:
        event (_type_): _description_
    """
    print(f"Block size CALLBACK: {event}")


def c_val_callback(event):
    """Callback function for the c value slider

    Args:
        event (_type_): _description_
    """
    print(f"C-value CALLBACK: {event}")


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


def opencving():
    window = ImageWindow()
    window.set_base_image(cv.imread("interactive ui\\cells8.tif"))
    window.open_window()


opencving()
