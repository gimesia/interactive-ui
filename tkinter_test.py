import tkinter as tk

import cv2 as cv
import skimage.io

from interactive_ui import *

# Read image
im = cv.imread("cells8.tif")
print(im)
print(im.shape)
# cv.imshow('Test', im)
# cv.waitKey(0)
# cv.destroyAllWindows()

app = App()
if is_open:
    app.model.set_base_image(im)
    app.view.refresh_photo()
    app.update_idletasks()
    app.update()
    app.listen()

app.mainloop()
app.dispose()
