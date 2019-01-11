import math
from skimage.io import imread, imsave
from skimage.filters import frangi
from skimage.filters import threshold_otsu
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.filters.rank import median
from skimage.morphology import disk

from matplotlib import pyplot

image = imread("testimage01.jpg", as_gray=True)
fig = pyplot.figure(dpi=250)
ax = fig.add_subplot(1, 1, 1)

edges = frangi(image, scale_range=(4, 5), beta1=0.1)
thresh = threshold_otsu(edges)
binary = edges > thresh
filled = ndimage.binary_fill_holes(binary)
de_noised = median(filled, disk(10))

label_img = label(de_noised)
regions = regionprops(label_img, coordinates='rc')
ax.imshow(image, 'gray')

for props in regions:
    y0, x0 = props.centroid
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.major_axis_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.major_axis_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.minor_axis_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.minor_axis_length
    x3 = x0 + math.sin(orientation) * 0.5 * props.minor_axis_length
    y3 = y0 + math.cos(orientation) * 0.5 * props.minor_axis_length
    x4 = x0 - math.cos(orientation) * 0.5 * props.major_axis_length
    y4 = y0 + math.sin(orientation) * 0.5 * props.major_axis_length

    ax.plot((x1, x4), (y1, y4), '-r', linewidth=0.5)
    ax.plot((x2, x3), (y2, y3), '-r', linewidth=0.5)
    ax.plot(x0, y0, '.g', markersize=5)

    print(props.eccentricity)
    print(props.centroid)
    print(props.orientation)
    print(props.local_centroid)
    break

ax.axis('off')
pyplot.show()
imsave("testimage01_center.png", de_noised)
