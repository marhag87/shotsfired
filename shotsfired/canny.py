import numpy as np
import matplotlib.pyplot as plt

from skimage import color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.io import imread


# Load picture and detect edges
image = imread("testimage01.jpg", as_gray=True)
edges = canny(image)

# Detect two radii
hough_radii = np.arange(20, 35, 2)
hough_res = hough_circle(edges, hough_radii)

# Select the most prominent 5 circles
accums, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=5)

# Draw them
fig, ax = plt.subplots()
image = color.gray2rgb(image)
for center_y, center_x, radius in zip(cy, cx, radii):
    circy, circx = circle_perimeter(center_y, center_x, radius)
    image[circy, circx] = (255, 0, 0)

ax.imshow(image)
plt.show()
