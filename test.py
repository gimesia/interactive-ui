# %%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2

# Generate random sample data
n_clusters = 4
cluster_data = [
    f'Cl_{np.random.randint(0, 4)}' for i in range(n_clusters)]
disabled_data = np.random.choice([True, False], size=n_clusters)
color_data_ = np.random.randint(0, 255, size=(n_clusters, 3))
area_data = np.random.randint(255, 2555, size=(n_clusters))
color_data = []
contours_data = []


# Generate random contours
for i in range(n_clusters):
    n_contours = np.random.randint(1, 5)
    cluster_contours = []
    for j in range(1):
        n_points = np.random.randint(4, 6)
        points = np.random.randint(10, 490, size=(n_points, 2))
        contour = cv2.convexHull(points)
        cluster_contours.append(contour)
    contours_data.append(cluster_contours)
    color_data.append(tuple(color_data_[i]))


# Create the DataFrame
df = pd.DataFrame({
    'Cluster': cluster_data,
    'Disabled': disabled_data,
    'Contours': contours_data,
    'Color': color_data,
    'Area': area_data
})
df


# %%
def draw_contours_on_image(df, img):
    """Draw contours for each row in a DataFrame on an input image.

    Args:
    df (pandas.DataFrame): DataFrame containing 'Contours' column with OpenCV contours.
    img (numpy.ndarray): Input image to draw contours on.

    Returns:
    numpy.ndarray: Image with contours drawn on it.
    """
    # Create a copy of the input image
    img_with_contours = img.copy()

    # Loop over each row in the DataFrame
    for i, contours in enumerate(df['Contours']):
        # Loop over each contour in the row
        for contour in contours:
            # Draw the contour on the image
            color = df["Color"][i]
            color = (int(color[0])/255, int(color[1])/255, int(color[2])/255)
            cv2.drawContours(img_with_contours, [
                             contour], -1, color, 1)

    return img_with_contours


# %%
im = np.zeros((500, 500, 3), float)


cv2.imshow("Image", draw_contours_on_image(df, im))
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
point = (220, 170)


def test_contour(contour):
    # assuming x and y are the coordinates of the point to test
    x, y = 100, 100  # replace with the coordinates of the point you want to test
    result = cv2.pointPolygonTest(contour[0], point, False)
    return result == 0.0 or result == 1.0


# %%
# Show result for each row
for i in df.iterrows():
    cnt = (i[1].Contours[0])
    print(cv2.pointPolygonTest(cnt, point, False))

# %%
# Perform polygon test on contours col
condition = df["Contours"].apply(
    lambda x: cv2.pointPolygonTest(x[0], point, False) >= 0
)
a = df.loc[condition]
df.loc[a.index, "Go"] = True
df

# %%
#
df


# %%
max = a["Area"].max()
condition2 = a["Area"].apply(lambda x: x == max)
b = a[condition2]
b.index
# %%
df[condition][condition2]
# %%
rows = df.loc[b.index]
rows
# %%
