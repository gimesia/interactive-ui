import numpy as np
import cv2 as cv


class Cluster():
    """ Class for a class of identified objects
    """

    def __init__(self, clustername: str, color):
        self.name = clustername
        self.color = color
        self.checked = False
        self.contours = []
        self.disabled_contours = []
        self.on_checked_changed = None

    def subscribe_on_checked_changed_event(self, func):
        self.on_checked_changed.append(func)

    def set_checked_state(self):
        self.checked = not self.checked
        if self.on_checked_changed is not None:
            self.on_checked_changed(self)

    def get_all_contours(self):
        a = np.concat(self.contours, self.disabled_contours)
        return a

    def disable_contour(self, index: int):
        print(f"Disabling contour at index #{index}")
        self.disabled_contours.append(self.contours.pop(index))

    def enable_contour(self, index: int):
        print(f"Enabling contour at index #{index}")
        self.contours.append(self.disabled_contours.pop(index))


class ImageWindow():
    def __init__(self):
        self.clusters: list[Cluster] = []
        self.og_img: np.ndarray = None
        self.contour_img: np.ndarray = None
        self.block_size = 101
        self.c_val = 21
        self.block_size_on_changed = []
        self.c_value_on_changed = []
        self.on_sliders_changed = []

        self.window = None

        self.subscribe_sliders_changed_event(self.calculate_adaptive_th)
        self.subscribe_sliders_changed_event(self.refresh_img)

    def open_window(self):
        self.window = 'Interactive UI'
        cv.namedWindow(self.window)
        cv.imshow(self.window, self.contour_img)
        cv.displayOverlay(
            self.window, "PRESS CTRL + P to modify shown clusters")
        # Sliders
        cv.createTrackbar('Block size:', self.window,
                          self.block_size, 255, self.set_block_size)
        cv.createTrackbar('C value:', self.window,
                          self.c_val, 255, self.set_c_value)

        cv.createButton('c1', button_callback, None, cv.QT_CHECKBOX, 1),
        cv.createButton('c2', button_callback, None, cv.QT_CHECKBOX, 2),
        cv.createButton('c3', button_callback, None, cv.QT_CHECKBOX, 3),
        cv.createButton('c4', button_callback, None, cv.QT_CHECKBOX, 4),

        cv.setMouseCallback(self.window, self.on_click)

        cv.waitKey(0)
        cv.destroyAllWindows()

    def set_base_image(self, img: np.ndarray):
        self.og_img = img.copy()
        self.contour_img = img.copy()

    def subscribe_sliders_changed_event(self, func):
        self.on_sliders_changed.append(func)

    def sliders_changed(self):
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
        self.sliders_changed()

    def set_c_value(self, val: int):
        if val < 1:
            val = 3

        print(f"@ C value set to: {val}")
        self.c_val = val
        self.sliders_changed()

    def calculate_adaptive_th(self):
        print("Calculating new adaptive threshold")

        gray_img = cv.cvtColor(self.og_img, cv.COLOR_BGR2GRAY)
        th = cv.adaptiveThreshold(
            gray_img, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, self.block_size, self.c_val)
        contours = cv.findContours(
            th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]

        # Divide found contours into clusters
        self.clusters = devide_contours_into_clusters(contours)

        return contours

    def refresh_img(self):
        print("Refreshing displayed img")
        self.contour_img = self.og_img.copy()

        # Drawing contours for clusters
        for cluster in self.clusters:
            # if not cluster.checked: continue
            cv.drawContours(self.contour_img, cluster.contours, -1,
                            color=cluster.color, thickness=-1, lineType=cv.LINE_8)
            cv.drawContours(self.contour_img, cluster.disabled_contours, -1,
                            color=cluster.color, thickness=1, lineType=cv.LINE_8)

        cv.imshow(self.window, self.contour_img)

    def on_click(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            print(f"@ Mouse click at {(x,y)}")
            poly_test((x, y), self.clusters)
            self.refresh_img()


# TODO: rethink
def button_callback(*args):
    print(f"Button clicked: {args}")


def poly_test(point: "tuple[int,int]", clusters: "list[Cluster]"):
    for cl in clusters:
        for j, cont in enumerate(cl.contours):
            if int(cv.pointPolygonTest(cont, point, False)) >= 0:
                print(f"{cl.name} cluster's #{j} element")
                cl.disable_contour(j)
                return
        for j, cont in enumerate(cl.disabled_contours):
            if int(cv.pointPolygonTest(cont, point, False)) >= 0:
                print(f"{cl.name} cluster's #{j} element")
                cl.enable_contour(j)


def devide_contours_into_clusters(contours):
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


def opencving():
    """Test of the code
    """
    window = ImageWindow()
    cluster = Cluster("Test", (0, 0, 255))
    window.clusters.append(cluster)
    window.set_base_image(cv.imread("interactive ui\\img\\cells8.tif"))
    window.open_window()


opencving()
