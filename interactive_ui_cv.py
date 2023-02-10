import cv2 as cv
import numpy as np
import pandas as pd

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
        self.disabled_contours.append(self.contours.pop(index))

    def enable_contour(self, index: int):
        self.contours.append(self.disabled_contours.pop(index))


class ObjectParams():
    def __init__(self, cluster_name, i, area, disabled):
        self.cluster = cluster_name
        self.indx = i
        self.area = area
        self.disabled = disabled

    def __str__(self) -> str:
        return f"cluster: {self.cluster}{' (disabled)' if self.disabled else ''}, index: {self.indx}, area: {self.area}"

    def put_params_on_image(self, img: np.ndarray, org: "tuple(int, int)", *kwargs):
        lines = list(map(lambda x: f"{x[0]}: {x[1]}", self.__dict__.items()))
        image = put_textbox_on_img(img, lines, (org[0]+10, org[1]))
        return image


class ImageWindow():
    def __init__(self, img=None):
        self.window = "Interactive UI"
        self.clusters: list[Cluster] = []
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.data: pd.DataFrame = None
        self.block_size = 101
        self.c_val = 21
        self.block_size_on_changed = []
        self.c_value_on_changed = []
        self.on_trackbar_changed = []
        self.edit = False
        self.preview = True
        self.select = False
        if img is not None:
            self.set_base_image(img)

        self.subscribe_trackbar_changed_event(self.calculate_clusters)
        self.subscribe_trackbar_changed_event(self.refresh_img)

    def timed_overlay_msg(self, text: str, time: int = 0):
        """Displays overlay for a defined time or permanently

        Args:
            text (str): Displayed text on overlay
            time (int, optional): Visibility time in seconds. Defaults to 0, meaning its permament.
        """
        cv.displayOverlay(self.window, f"{text}", time * 1000)

    def timed_statusbar_msg(self, text: str, time: int = 0):
        """Displays overlay for a defined time or permanently

        Args:
            text (str): Displayed text on overlay
            time (int, optional): Visibility time in seconds. Defaults to 0, meaning its permament.
        """
        cv.displayStatusBar(self.window, f"{text}", time * 1000)

    def get_claster_by_name(self, cluster_name: str) -> Cluster:
        index = list(map(lambda x: x.name, self.clusters)).index(cluster_name)
        return self.clusters[index]

    def open_window(self):
        """Opens window with segemented image and controls
        """
        cv.namedWindow(self.window)
        cv.imshow(self.window, self.contour_img)

        # Sliders
        cv.createTrackbar('Block size:', self.window,
                          self.block_size, 255, self.set_block_size)
        cv.createTrackbar('C value:', self.window,
                          self.c_val, 255, self.set_c_value)

        # Mouse event callbacks
        cv.setMouseCallback(self.window, self.on_mouse_event)

        # Finding clusters
        self.calculate_clusters()

        # Creating control buttons
        cv.createButton("INFO MODE", self.on_toggle_mode,
                        "INFO", cv.QT_RADIOBOX, not self.edit)
        cv.createButton("EDIT MODE", self.on_toggle_mode, "EDIT",
                        cv.QT_RADIOBOX, self.edit)
        self.create_cluster_checkboxes()
        cv.createButton("EN/DISABLE ALL", self.disable_all, "ALL",
                        cv.QT_PUSH_BUTTON | cv.QT_NEW_BUTTONBAR, 0)
        cv.createButton("DISPLAY STATS", self.display_stats, "DISPLAY",
                        cv.QT_PUSH_BUTTON | cv.QT_NEW_BUTTONBAR, 0)
        cv.createButton("SAVE DATA", self.save_data, "print",
                        cv.QT_PUSH_BUTTON)

        # Close when ESC is pressed
        while True:
            k = cv.waitKey(0) & 0xFF
            if k == 27:
                cv.destroyAllWindows()
                break

    def disable_all(self, *args):
        next = not self.preview
        self.preview = next
        self.timed_overlay_msg(
            f"Clusters toggled {'on' if self.preview else 'off'}", 2)
        self.timed_statusbar_msg(
            f"Clusters toggled {'on' if self.preview else 'off'}", 2)
        self.refresh_img()

    def set_base_image(self, img: np.ndarray):
        self.og_img = img.copy()
        self.contour_img = img.copy()

    def subscribe_trackbar_changed_event(self, func):
        self.on_trackbar_changed.append(func)

    def set_block_size(self, val: int):
        # Has to be blocksize % 2 == 1 && blocksize % 2 > 1
        if val < 3:
            val = 3
        if val % 2 != 1:
            val += 1

        terminal_text.event(f"@ Block size set to: {val}")
        self.block_size = val
        self.on_trackbar_sliders_changed()

    def set_c_value(self, val: int):
        if val < 1:
            val = 3

        terminal_text.event(f"@ C value set to: {val}")
        self.c_val = val
        self.on_trackbar_sliders_changed()

    def calculate_clusters(self) -> np.ndarray:
        """Calculates the adaptive threshold and clusters for the src image with the selected C and Blocksize values

        Returns:
            np.ndarray: array of contours
        """
        terminal_text.event("Calculating new adaptive threshold")
        checkbox_values = list(map(lambda x: x.checked, self.clusters))
        self.clusters = segmentation(self.og_img, self.block_size, self.c_val)
        for i, val in enumerate(checkbox_values):
            self.clusters[i].checked = val

        return self.clusters

    @debounce(0.025)
    def refresh_img(self):
        """Draws contours again on original image, according to the object's parameters
        """
        terminal_text.warn("@ Refreshing displayed img")
        self.contour_img = self.og_img.copy()

        # Drawing contours for clusters
        if self.preview:
            for cluster in self.clusters:
                if not cluster.checked:
                    continue  # Exits iteration if cluster is not checked
                cv.drawContours(self.contour_img, cluster.contours, -1,
                                color=cluster.color, thickness=1, lineType=cv.LINE_4)
                cv.drawContours(self.contour_img, cluster.disabled_contours, -1,
                                color=(150, 150, 150), thickness=0, lineType=cv.LINE_4)

        cv.imshow(self.window, self.contour_img)

    def create_cluster_checkboxes(self):
        """Dynamically create checkbox controls for disabling the visibility of clusters
        """
        for cl in self.clusters:
            cv.createButton(cl.name, self.on_toggle_cluster, cl.name,
                            cv.QT_CHECKBOX | cv.QT_NEW_BUTTONBAR, cl.checked)

    def save_data(self, *args):
        self.timed_overlay_msg("Saving results", 3)
        current_data = self.extract_data()
        if self.data is None:
            current_data.to_csv("original_data.csv")
        else:
            current_data.to_csv("modified_data.csv")
            self.data.to_csv("original_data.csv")
        stats = self.extract_stats()
        stats.to_csv("stats.csv")

    def extract_data(self, disableds=False):
        """Creates dataframe from active contours of the clusters

        Returns:
            pandas.DataFrame: Dataframe with object properties
        """
        df = pd.DataFrame({"cluster": [], "index": [], "area": []})
        for i, cluster in enumerate(self.clusters):
            if not cluster.checked:
                continue
            for j, contour in enumerate(self.clusters[i].contours):
                # Enchance peformance wise
                df.loc[len(df.index)] = [
                    cluster.name, j, cv.contourArea(contour)]
            if disableds:
                for j, contour in enumerate(self.clusters[i].disabled_contours):
                    # Enchance peformance wise
                    df.loc[len(df.index)] = [
                        cluster.name, j, cv.contourArea(contour)]
        return df

    def extract_stats(self, *args) -> pd.DataFrame:
        """Extracts the summary of the clusters' distributions
        """
        lengths = np.asarray(
            list(map(lambda x: len(x.contours), self.clusters)))
        disableds = np.sum(np.asarray(
            list(map(lambda x: len(x.disabled_contours), self.clusters))))
        all = np.sum(lengths)
        indexes = list(map(lambda x: x.name, self.clusters))

        frame = {'Count': pd.Series(lengths, index=indexes),
                 'Percentage': pd.Series((lengths / all) * 100, index=indexes)}
        df1 = pd.DataFrame(frame, index=indexes)
        df2 = pd.DataFrame({'Count': [all, disableds], 'Percentage': [
            100, 0]}, index=['all', 'disabled'])
        return pd.concat(objs=[df1, df2])

    def display_stats(self, *args):
        """Displays cluster statistics on window
        """
        image = self.contour_img.copy()
        stats = self.extract_stats()

        lines = []

        for index, row in stats.iterrows():
            lines.append(
                f"{index}: {row['Count']} ({round(row['Percentage'], 2)}%)")
        image = put_textbox_on_img(image, lines, (25, 25))
        cv.imshow(self.window, image)

    def on_toggle_mode(self, val, mode):
        """Toggles windows info <-> edit mode

        Args:
            val (int): payload of the button event
            mode (str): mode flag of the button event
        """
        if mode == "EDIT":
            text = f"{'EDIT' if val else 'INFO' } MODE"
            self.timed_overlay_msg(text, 3)
            self.timed_statusbar_msg(text, 1)
            self.edit = bool(val)
            if val and self.data is None:
                terminal_text.event("Storing initial cluster data")
                self.data = self.extract_data()
        else:
            return

    def on_trackbar_sliders_changed(self):
        terminal_text.event(f"@ Trackbar sliders changed")
        if self.edit:
            self.on_toggle_mode(0, "EDIT")
            self.data = None

        for function in self.on_trackbar_changed:
            function()

    @debounce(0.05)
    def on_mouse_event(self, event, x: int, y: int, *args):
        """Mouse event listener with all the respective actions to be listened to (click, dblclick, hover, etc.)

        Args:
            event (str): type of the mouse event
            x (int): x coord
            y (int): y coord
        """
        point = (x, y)

        if not self.edit:
            obj_params = find_object_for_point(point, self.clusters)
            if event == cv.EVENT_LBUTTONDOWN:
                if obj_params is not None:
                    cl = self.get_claster_by_name(obj_params.cluster)
                    img = self.contour_img.copy()
                    img = obj_params.put_params_on_image(img, point)
                    img = cv.drawContours(img, cl.contours[obj_params.indx], -1,
                                          color=(0, 0, 0), thickness=-2, lineType=cv.LINE_8)

                if obj_params is not None:
                    self.timed_statusbar_msg(f"{obj_params.__str__()}")
                    cv.imshow(self.window, img)
                else:
                    cv.imshow(self.window, self.contour_img)
                    self.timed_statusbar_msg(f"No object at given point", 1)

        if self.edit:
            # Left btn click disables clicked object
            if event == cv.EVENT_LBUTTONDOWN and not self.select:
                result = find_object_for_point(point, self.clusters, True)
                if result:
                    self.timed_statusbar_msg(f"{result.__str__()}")
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

    def on_toggle_cluster(self, payload, cluster_name, *args):
        self.get_claster_by_name(cluster_name).checked = bool(payload)
        self.refresh_img()


def find_object_for_point(point: "tuple[int,int]", clusters: "list[Cluster]", disable=False) -> ObjectParams or None:
    """Checks whether given point is in any of the given clusters' objects

    Args:
        point (tuple[int,int]): coordinates of point
        clusters (list[Cluster]): clusters to be searched in
        disable (bool): flag whether function should dis-/enable found object
    """
    for cluster in clusters:
        if not cluster.checked:
            continue
        for j, contour in enumerate(cluster.contours):
            # If point is on or inside the contour
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:
                # Move to disabled list if disabling is turned on
                if disable:
                    terminal_text.event(
                        f"Disabling {cluster.name} cluster's #{j} object")
                    cluster.disable_contour(j)
                    return ObjectParams(cluster.name, j, cv.contourArea(contour), True)
                return ObjectParams(cluster.name, j, cv.contourArea(contour), False)

        for j, contour in enumerate(cluster.disabled_contours):
            # If point is on or inside the contour
            if int(cv.pointPolygonTest(contour, point, False)) >= 0:
                # Move to enabled list if disabling is turned on
                if disable:
                    terminal_text.event(
                        f"Enabling {cluster.name} cluster's #{j} object")
                    cluster.enable_contour(j)
                    return ObjectParams(cluster.name, j, cv.contourArea(contour), False)
                return ObjectParams(cluster.name, j, cv.contourArea(contour), True)
    return None


# NOTE: This is function should be replace with a more precise classification
def divide_contours_into_clusters(contours: np.ndarray, ) -> "list[Cluster]":
    """Classifies contours (polygons) into clusters given clusters
        Note: currently based on size
    Args:
        contours (_type_): polygons

    Returns:
        list[Cluster]: clusters
    """
    clusters: "list[Cluster]" = [Cluster("c1", (244, 133, 66)), Cluster(
        "c2", (83, 168, 52)), Cluster("c3", (5, 188, 251)), Cluster("c4", (53, 67, 234))]
    # NOTE: hardcoded clusters!

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
        if areas[i] < 1.0:
            continue
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
    return divide_contours_into_clusters(contours)


def put_text(img: np.ndarray, text: str, org: "tuple(int, int)", font=cv.FONT_HERSHEY_PLAIN,
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
    im = cv.putText(img.copy(), text, org, font, fontScale,
                    colors[0], thickness, cv.LINE_AA)
    im = cv.putText(im, text, org, font, fontScale,
                    colors[1], 1, cv.LINE_AA)
    return im


def put_textbox_on_img(img, lines: "list[str]", point):
    image = img.copy()
    end_point = (point[0] + 200, point[1] + len(lines) * 30)

    diff = (img.shape[1]-end_point[0], img.shape[0]-end_point[1])
    if diff[0] < 0:
        point = (point[0]+diff[0], point[1])
        end_point = (end_point[0]+diff[0], end_point[1])
    if diff[1] < 0:
        point = (point[0], point[1]+diff[1])
        end_point = (end_point[0], end_point[1]+diff[1])

    image = cv.rectangle(
        image, point, end_point, (0, 0, 0), 3)
    image = cv.rectangle(
        image, point, end_point, (225, 225, 225), -1)
    point = (point[0] + 10, point[1] + 20)
    for line in lines:
        image = put_text(image, str(line), point)
        point = (point[0], point[1] + 30)
    return image


def opencving():
    """Test of the code
    """
    window = ImageWindow(cv.imread("img\\cells8.tif"))
    window.open_window()


opencving()
