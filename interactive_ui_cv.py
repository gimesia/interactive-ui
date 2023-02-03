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


class ObjectParams():
    def __init__(self, cluster_name, i, area):
        self.cluster = cluster_name
        self.indx = i
        self.area = area

    def put_params_on_window(self, window, img: np.ndarray, org: "tuple(int, int)", *kwargs):
        y0, dy = org[1], 25
        for i, (key, val) in enumerate(self.__dict__.items()):
            y = y0 + i*dy
            img = put_bordered_text(
                img, f"{key}: {val}", (org[0], y))
        cv.imshow(window, img)
        return img


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
        self.edit = False
        self.select = True
        self.selected = None
        if img is not None:
            self.set_base_image(img)

        self.subscribe_trackbar_changed_event(self.calculate_clusters)
        self.subscribe_trackbar_changed_event(self.refresh_img)

    def timed_overlay(self, text: str, time: int = 0):
        """Displays overlay for a defined time or permanently

        Args:
            text (str): Displayed text on overlay
            time (int, optional): Visibility time in seconds. Defaults to 0, meaning its permament.
        """
        print(f"overlay: {text}, {time}")
        cv.displayOverlay(
            self.window, f"{text}", time * 10)

    def get_claster_by_name(self, cluster_name: str) -> Cluster:
        index = list(map(lambda x: x.name, self.clusters)).index(cluster_name)
        return self.clusters[index]

    def open_window(self):
        """Opens window with segemented image and controls
        """
        cv.namedWindow(self.window)
        cv.imshow(self.window, self.contour_img)

        # self.timed_overlay("PRESS CTRL + P to open properties window", 10)
        # cv.displayOverlay(self.window, )

        # Sliders
        cv.createTrackbar('Block size:', self.window,
                          self.block_size, 255, self.set_block_size)
        cv.createTrackbar('C value:', self.window,
                          self.c_val, 255, self.set_c_value)

        # Button toggling edit mode
        cv.createButton("EDIT MODE", self.on_toggle_edit_mode, "EDIT",
                        cv.QT_CHECKBOX | cv.QT_NEW_BUTTONBAR, self.edit)
        cv.createButton("SELECT MODE", self.on_toggle_select_mode,
                        "SELECT", cv.QT_CHECKBOX, self.select)

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
        """Draws contours again on original image, according to the object's parameters
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
                            color=cluster.color, thickness=-1, lineType=cv.LINE_4)
            cv.drawContours(self.contour_img, cluster.disabled_contours, -1,
                            color=cluster.color, thickness=1, lineType=cv.LINE_4)

        cv.imshow(self.window, self.contour_img)

    def create_cluster_checkboxes(self):
        """Dynamically create checkbox controls for disabling the visibility of clusters
        """
        for cl in self.clusters:
            cv.createButton(cl.name, self.on_toggle_cluster, cl.name,
                            cv.QT_CHECKBOX | cv.QT_NEW_BUTTONBAR, cl.checked)

    @debounce(0.05)
    def on_mouse_event(self, event, x: int, y: int, *args):
        """Mouse event listener with all the respective actions to be listened to (click, dblclick, hover, etc.)

        Args:
            event (str): type of the mouse event
            x (int): x coord
            y (int): y coord
        """
        point = (x, y)

        if self.select:
            if event == cv.EVENT_LBUTTONDOWN:
                obj_params = find_object_for_point(point, self.clusters)
                if obj_params is not None:
                    cl = self.get_claster_by_name(obj_params.cluster)

                    img = cv.drawContours(self.contour_img.copy(), cl.contours[obj_params.indx], -1,
                                          color=(255-cl.color[0], 255-cl.color[1], 255-cl.color[2]), thickness=2, lineType=cv.LINE_4)
                    img = obj_params.put_params_on_window(
                        self.window, img, point)
                    cv.imshow(self.window, img)
                else:
                    cv.imshow(self.window, self.contour_img)

        if self.edit:
            # Left btn click disables clicked object
            if event == cv.EVENT_LBUTTONDOWN and not self.select:
                terminal_text.event(
                    f"@ Left click at {point} -> Searching for object in displayed clusters")
                result = find_object_for_point(point, self.clusters, True)
                if result:
                    self.refresh_img()
                return

            # Middle btn clicks moves object to next cluster in list NOTE: Even if the cluster is not displayed (via the checkboxes)
            elif event == cv.EVENT_MBUTTONDOWN:
                for i, cluster in enumerate(self.clusters):
                    if not cluster.checked:  # Skips disabled clusters
                        continue
                    for j, contour in enumerate(cluster.contours):
                        if cv.pointPolygonTest(contour, point, False) >= 0:
                            mod = len(self.clusters)
                            next_i = (i + 1) % mod

                            terminal_text.event(
                                f"@ Middle click at {point} -> Changing cluster of object from {cluster.name} to {self.clusters[next_i].name}")
                            # Moving object to the next cluster
                            self.clusters[next_i].contours.append(
                                cluster.contours.pop(j))
                            self.refresh_img()
                            return
                return

    def on_toggle_edit_mode(self, payload, *args):
        txt = f"Edit mode: {bool(not self.edit)}"
        terminal_text.event(txt)
        self.edit = bool(payload)
        self.timed_overlay(txt, 1)

    def on_toggle_select_mode(self, payload, *args):
        txt = f"Select mode: {bool(payload)}"
        terminal_text.event(txt)
        self.select = bool(payload)
        self.timed_overlay(txt, 1)

    def on_toggle_cluster(self, payload, cluster_name, *args):
        self.get_claster_by_name(cluster_name).checked = bool(payload)
        self.refresh_img()


def find_object_for_point(point: "tuple[int,int]", clusters: "list[Cluster]", disable=False) -> ObjectParams or None:
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
            # If point is on or inside the contour
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:
                print(f"{cluster.name} cluster's #{j} element")
                # Move to disabled list if disabling is turned on
                if disable:
                    cluster.disable_contour(j)
                return ObjectParams(cluster.name, j, cv.contourArea(contour))

        for j, contour in enumerate(cluster.disabled_contours):
            # If point is on or inside the contour
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:
                print(f"{cluster.name} cluster's #{j} element")
                # Move to enabled list if disabling is turned on
                if disable:
                    cluster.enable_contour(j)
                return ObjectParams(cluster.name, j, cv.contourArea(contour))
    return None


# NOTE: This is function should be replace with a more precise classification
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


# NOTE: This is function should be replace with a more precise segmentation
def segmentation(img: np.ndarray, block_size: int, c_val: int):
    """Object segmention on image

    Args:
        img (np.ndarray): _description_
        block_size (_type_): _description_
        c_val (_type_): _description_

    Returns:
        _type_: _description_
    """
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    th = cv.adaptiveThreshold(
        gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, block_size, c_val)
    contours = cv.findContours(
        th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    return devide_contours_into_clusters(contours)


def put_bordered_text(img: np.ndarray, text: str, org: "tuple(int, int)", font=cv.FONT_HERSHEY_PLAIN,
                      fontScale: int = 1, colors: "tuple(tuple(int, int, int), tuple(int, int, int))" = ((0, 0, 0), (255, 255, 255)), thickness: int = 3) -> np.ndarray:
    """Draws bordered text on image

    Args:
        img (np.ndarray): target image
        text (str): drawn text
        org (tuple): starting coordinate of the text
        font (_type_, optional): fontstyle. Defaults to cv.FONT_HERSHEY_SIMPLEX.
        fontScale (int, optional): fontscale. Defaults to 1.
        colors (tuple, optional): colors of the text. Defaults to ((0, 0, 0), (255, 255, 255)).
        thickness (int, optional): thickness. Defaults to 2.

    Returns:
        np.ndarray: image with text
    """
    img = cv.putText(img, text, org, font, fontScale,
                     colors[0], thickness, cv.LINE_AA)
    img = cv.putText(img, text, org, font, fontScale,
                     colors[1], 1, cv.LINE_AA)
    return img


def opencving():
    """Test of the code
    """
    window = ImageWindow(cv.imread("img\\cells8.tif"))
    window.open_window()


opencving()
