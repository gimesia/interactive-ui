import numpy as np
import cv2 as cv

from debounce import debounce
import terminal_text  # only for debugging


class Cluster():
    """ Class for a class of identified objects
    """

    def __init__(self, clustername: str, color):
        self.name = clustername
        self.color = color
        self.checked = True
        self.contours = []
        self.disabled_contours = []

    def get_all_contours(self):
        return np.concat(self.contours, self.disabled_contours)

    def disable_contour(self, index: int):
        print(f"Disabling contour at index #{index}")
        self.disabled_contours.append(self.contours.pop(index))

    def enable_contour(self, index: int):
        print(f"Enabling contour at index #{index}")
        self.contours.append(self.disabled_contours.pop(index))


class ImageWindow():
    def __init__(self, img=None):
        self.window = "Interactive UI"
        self.clusters: list[Cluster] = []
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.block_size = 101
        self.c_val = 21
        self.block_size_on_changed = []
        self.c_value_on_changed = []
        self.on_sliders_changed = []
        self.edit = True
        if img is not None:
            self.set_base_image(img)

        self.subscribe_trackbar_changed_event(self.calculate_clusters)
        self.subscribe_trackbar_changed_event(self.refresh_img)

    def open_window(self):
        """Opens window with segemented image and controls
        """
        cv.namedWindow(self.window)
        cv.imshow(self.window, self.contour_img)

        cv.displayOverlay(
            self.window, "PRESS CTRL + P to open ")

        # Sliders
        cv.createTrackbar('Block size:', self.window,
                          self.block_size, 255, self.set_block_size)
        cv.createTrackbar('C value:', self.window,
                          self.c_val, 255, self.set_c_value)

        # Button toggling edit mode
        cv.createButton("EDIT MODE", self.on_toggle_edit_mode, "EDIT",
                        cv.QT_RADIOBOX | cv.QT_NEW_BUTTONBAR)
        cv.createButton("SELECT MODE", self.on_toggle_edit_mode,
                        "SELECT", cv.QT_RADIOBOX)

        # mouse event callbacks
        cv.setMouseCallback(self.window, self.on_mouse_event)

        self.calculate_clusters()
        self.create_cluster_checkboxes()

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
        terminal_text.event(f"@ Sliders changed")
        for function in self.on_sliders_changed:
            function()

    def set_block_size(self, val: int):
        # Has to be blocksize % 2 == 1 && blocksize % 2 > 1
        if val < 3:
            val = 3
        if val % 2 != 1:
            val += 1

        terminal_text.event(f"@ Block size set to: {val}")
        self.block_size = val
        self.trackbar_changed()

    def set_c_value(self, val: int):
        if val < 1:
            val = 3

        terminal_text.event(f"@ C value set to: {val}")
        self.c_val = val
        self.trackbar_changed()

    def calculate_clusters(self) -> np.ndarray:
        """Calculates the adaptive threshold and clusters for the src image with the selected C and Blocksize values

        Returns:
            np.ndarray: array of contours
        """
        print("Calculating new adaptive threshold")
        vals = list(map(lambda x: x.checked, self.clusters))
        self.clusters = segmentation(self.og_img, self.block_size, self.c_val)
        for i, val in enumerate(vals):
            self.clusters[i].checked = val
        return self.clusters

    @debounce(0.025)
    def refresh_img(self):
        """Draws contours again on original image, according to the given parameters
        """
        terminal_text.warn("@ Refreshing displayed img")
        self.contour_img = self.og_img.copy()

        # Drawing contours for clusters
        for cluster in self.clusters:
            # print(f"{cluster.checked}")
            # print(f"{cluster.name} is turned {'on' if cluster.checked else 'off'}")
            if not cluster.checked:
                continue  # Exits iteration if cluster is not checked
            cv.drawContours(self.contour_img, cluster.contours, -1,
                            color=cluster.color, thickness=-1, lineType=cv.LINE_8)
            cv.drawContours(self.contour_img, cluster.disabled_contours, -1,
                            color=cluster.color, thickness=1, lineType=cv.LINE_8)

        cv.imshow(self.window, self.contour_img)

    def create_cluster_checkboxes(self):
        for cl in self.clusters:
            cv.createButton(cl.name, self.on_toggle_cluster, cl.name,
                            cv.QT_CHECKBOX | cv.QT_NEW_BUTTONBAR, cl.checked)

    def on_toggle_edit_mode(self, *args):
        terminal_text.event(f"Edit mode: {bool(not self.edit)}")
        self.edit = not self.edit

    @debounce(0.05)
    def on_mouse_event(self, event, x: int, y: int, flags, *args):
        """Mouse event listener with all the respective actions to be listened to (click, dblclick, hover, etc.)

        Args:
            event (str): type of the mouse event
            x (int): x coord
            y (int): y coord
        """
        point = (x, y)

        if self.edit:
            if event == cv.EVENT_LBUTTONDOWN:
                terminal_text.event(
                    f"@ LDblClick at {point} -> Searching for object in displayed clusters")
                result = find_cluster_for_point(point, self.clusters)
                if result:
                    self.refresh_img()

            elif event == cv.EVENT_MBUTTONDOWN:
                for i, cluster in enumerate(self.clusters):
                    if not cluster.checked:
                        continue
                    for j, contour in enumerate(cluster.contours):
                        result = int(cv.pointPolygonTest(
                            contour, point, False)) >= 0
                        if result:
                            mod = len(self.clusters)
                            next_i = (i + 1) % mod

                            terminal_text.event(
                                f"@ MiddleClick at {point} -> Changing cluster of object from {cluster.name} to {self.clusters[next_i].name}")

                            self.clusters[next_i].contours.append(
                                cluster.contours.pop(j))
                            self.refresh_img()
                            return

    def on_toggle_cluster(self, *args):
        terminal_text.event(f"@ Toggle {args[1]} {bool(args[0])}")
        index = list(map(lambda x: x.name, self.clusters)).index(args[1])
        self.clusters[index].checked = bool(args[0])
        self.refresh_img()


def find_cluster_for_point(point: "tuple[int,int]", clusters: "list[Cluster]") -> bool:
    """Checks whether given point is in any of the given clusters' objects

    Args:
        point (tuple[int,int]): coordinates of point
        clusters (list[Cluster]): clusters to be searched in
    """
    print("Findig cluster for point")
    for cluster in clusters:
        if not cluster.checked:
            continue
        for j, contour in enumerate(cluster.contours):
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:
                print(f"{cluster.name} cluster's #{j} element")
                cluster.disable_contour(j)
                return True
        for j, contour in enumerate(cluster.disabled_contours):
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:
                print(f"{cluster.name} cluster's #{j} element")
                cluster.enable_contour(j)
                return True
    return False


# TODO make it dynamic somehow (hardcoded clusters => bad)
def devide_contours_into_clusters(contours) -> "list[Cluster]":
    """Classifies polygons into clusters based on size

    Args:
        contours (_type_): polygons

    Returns:
        list[Cluster]: clusters
    """
    print("Sorting contours (by area)")

    # NOTE: hardcoded clusters!
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


def segmentation(img: np.ndarray, block_size, c_val):
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    th = cv.adaptiveThreshold(
        gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block_size, c_val)
    contours = cv.findContours(
        th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    return devide_contours_into_clusters(contours)


def opencving():
    """Test of the code
    """
    window = ImageWindow(cv.imread("img\\cells8.tif"))
    window.open_window()


opencving()
