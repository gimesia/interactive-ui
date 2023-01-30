import numpy as np
import cv2 as cv


class Cluster():
    """ Class for a class of identified objects
    """

    def __init__(self, clustername: str, color):
        self.name = clustername
        self.color = color
        self.checked = True
        self.contours = []
        self.disabled_contours = []
        self.on_checked_changed = []

    def subscribe_on_checked_changed_event(self, func):
        self.on_checked_changed.append(func)

    def toggle_checked_state(self, payload, name):
        print(f"{name} was toggled: {payload}")
        print(f"checked: {self.checked}")
        self.checked = payload
        print(f"checked: {self.checked}")
        for func in self.on_checked_changed:
            func()

    def get_all_contours(self):
        return np.concat(self.contours, self.disabled_contours)

    def disable_contour(self, index: int):
        print(f"Disabling contour at index #{index}")
        self.disabled_contours.append(self.contours.pop(index))

    def enable_contour(self, index: int):
        print(f"Enabling contour at index #{index}")
        self.contours.append(self.disabled_contours.pop(index))


class ImageWindow():
    def __init__(self):
        self.window = "Interactive UI"
        self.clusters: list[Cluster] = []
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.block_size = 101
        self.c_val = 21
        self.block_size_on_changed = []
        self.c_value_on_changed = []
        self.on_sliders_changed = []
        self.edit = False

        self.subscribe_trackbar_changed_event(self.calculate_adaptive_th)
        self.subscribe_trackbar_changed_event(self.refresh_img)

    def open_window(self):
        cv.namedWindow(self.window)
        cv.imshow(self.window, self.contour_img)
        cv.displayOverlay(
            self.window, "PRESS CTRL + P to open ")
        # Sliders
        cv.createTrackbar('Block size:', self.window,
                          self.block_size, 255, self.set_block_size)
        cv.createTrackbar('C value:', self.window,
                          self.c_val, 255, self.set_c_value)

        # Has to be dynamic, needs to follow number of clusters
        cv.createButton(
            "c1", self.clusters[0].toggle_checked_state, "c1", cv.QT_CHECKBOX | cv.QT_NEW_BUTTONBAR, 1),
        cv.createButton(
            "c2", self.clusters[1].toggle_checked_state, "c2", cv.QT_CHECKBOX | cv.QT_NEW_BUTTONBAR, 1),
        cv.createButton(
            "c3", self.clusters[2].toggle_checked_state, "c3", cv.QT_CHECKBOX | cv.QT_NEW_BUTTONBAR, 1),
        cv.createButton(
            "c4", self.clusters[3].toggle_checked_state, "c4", cv.QT_CHECKBOX | cv.QT_NEW_BUTTONBAR, 1),

        # create a push button in a new row
        cv.createButton("button6", callbackButton, None,
                        cv.QT_PUSH_BUTTON | cv.QT_NEW_BUTTONBAR)

        cv.setMouseCallback(self.window, self.on_click)

        # Only close when ESC is pressed
        while True:
            k = cv.waitKey(0) & 0xFF
            if k == 27:
                cv.destroyAllWindows()
                break

    def set_base_image(self, img: np.ndarray):
        self.og_img = img.copy()
        self.contour_img = img.copy()

    def subscribe_trackbar_changed_event(self, func):
        self.on_sliders_changed.append(func)

    def trackbar_changed(self):
        """Fires stored functions when a trackbar changes value
        """
        print(f"@ Sliders changed")
        for function in self.on_sliders_changed:
            function()

    def set_block_size(self, val: int):
        # Has to be blocksize % 2 == 1 && blocksize % 2 > 1
        if val < 3:
            val = 3
        if val % 2 != 1:
            val += 1

        print(f"@ Block size set to: {val}")
        self.block_size = val
        self.trackbar_changed()

    def set_c_value(self, val: int):
        if val < 1:
            val = 3

        print(f"@ C value set to: {val}")
        self.c_val = val
        self.trackbar_changed()

    def calculate_adaptive_th(self):
        """Calculates the adaptive threshold and clusters for the src image with the selected C and Blocksize values

        Returns:
            np.ndarray: array of contours
        """
        print("Calculating new adaptive threshold")

        gray_img = cv.cvtColor(self.og_img, cv.COLOR_BGR2GRAY)
        th = cv.adaptiveThreshold(
            gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, self.block_size, self.c_val)
        contours = cv.findContours(
            th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        # Divide found contours into clusters
        self.clusters = devide_contours_into_clusters(contours)
        for cl in self.clusters:
            cl.subscribe_on_checked_changed_event(self.refresh_img)

        return contours

    def refresh_img(self):
        """Draws contours again on original image, according to the given parameters
        """
        print("Refreshing displayed img")
        self.contour_img = self.og_img.copy()

        # Drawing contours for clusters
        for cluster in self.clusters:
            print(f"{cluster.checked}")
            print(f"{cluster.name} is turned {'on' if cluster.checked else 'off'}")
            if not cluster.checked:
                continue  # Exits iteration if cluster is not checked
            cv.drawContours(self.contour_img, cluster.contours, -1,
                            color=cluster.color, thickness=-1, lineType=cv.LINE_8)
            cv.drawContours(self.contour_img, cluster.disabled_contours, -1,
                            color=cluster.color, thickness=1, lineType=cv.LINE_8)

        cv.imshow(self.window, self.contour_img)

    def on_click(self, event, x: int, y: int, *args):
        """Mouse event listener with all the respective actions to be listened to (click, dblclick, hover, etc.)

        Args:
            event (str): type of the mouse event
            x (int): x coord
            y (int): y coord
        """

        if event == cv.EVENT_LBUTTONDOWN:
            print(f"@ Mouse click at {(x,y)}")
            find_cluster_for_point((x, y), self.clusters)
            self.refresh_img()


def find_cluster_for_point(point: "tuple[int,int]", clusters: "list[Cluster]"):
    """Checks whether given point is in any of the given clusters' objects

    Args:
        point (tuple[int,int]): coordinates of point
        clusters (list[Cluster]): clusters to be searched in
    """
    for cl in clusters:
        if not cl.checked:
            continue
        for j, cont in enumerate(cl.contours):
            if int(cv.pointPolygonTest(cont, point, False)) >= 0:
                print(f"{cl.name} cluster's #{j} element")
                cl.disable_contour(j)
                return
        for j, cont in enumerate(cl.disabled_contours):
            if int(cv.pointPolygonTest(cont, point, False)) >= 0:
                print(f"{cl.name} cluster's #{j} element")
                cl.enable_contour(j)


# TODO make it dynamic somehow (hardcoded clusters => bad)
def devide_contours_into_clusters(contours) -> "list[Cluster]":
    """Classifies polygons into clusters based on size

    Args:
        contours (_type_): polygons

    Returns:
        list[Cluster]: clusters
    """
    print("Sorting contours (by area)")

    clusters = [Cluster("c1", (244, 133, 66)), Cluster(
        "c2", (83, 168, 52)), Cluster("c3", (5, 188, 251)), Cluster("c4", (53, 67, 234))]

    # List of the areas of countours (matching indices)
    areas = list(map(lambda x: cv.contourArea(x), contours))

    mn = np.floor(np.min(areas))
    mx = np.ceil(np.max(areas))

    range = mx - mn
    quarter = range / 4

    # Calculating area limits for clusters
    c1_limit, c2_limit, c3_limit = mn + \
        quarter, mn + (1.5 * quarter), mn + (2.5 * quarter)

    for i, cont in enumerate(contours):
        if areas[i] < c1_limit:
            clusters[0].contours.append(cont)
        elif areas[i] < c2_limit:
            clusters[1].contours.append(cont)
        elif areas[i] < c3_limit:
            clusters[2].contours.append(cont)
        else:
            clusters[3].contours.append(cont)

    return clusters


def callbackButton(event, val):
    print("BUTTON CALLBACK")
    print(f"Event: {event}")
    print(f"Value: {val}")
    print(f"{cv.loadWindowParameters('Interactive UI')}")


def opencving():
    """Test of the code
    """
    window = ImageWindow()
    window.set_base_image(cv.imread("img\\cells8.tif"))
    window.open_window()


opencving()
