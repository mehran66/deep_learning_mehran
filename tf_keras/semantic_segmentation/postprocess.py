import skimage
from skimage import measure, io, img_as_ubyte
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage.morphology import opening, dilation, erosion
import pandas as pd
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import find_contours, approximate_polygon
from skimage.color import label2rgb

def measure_mask(image, mask, probability=True):

    if probability:
        # apply Gaussian blur, creating a new image
        sigma = 1
        blurred = skimage.filters.gaussian(mask, sigma=(sigma, sigma), truncate=3.5)

        # fine optimum threshold
        threshold = find_threshold(blurred)

        mask = blurred > threshold


    # Morphological operations to remove small noise - opening
    kernel = np.ones((10, 10), np.uint8)
    open_img = opening(mask, kernel)

    label_image, nbr_polys = measure.label(open_img, connectivity=mask.ndim, return_num=True)

    #image_label_overlay = label2rgb(label_image, image=image)


    # measure properties
    # area is the number of pixels in version 0.19.x, in the next version num_pixels would be added and area would be corrected
    props = measure.regionprops_table(label_image, image,
                              properties=['label','area', 'bbox', 'coords', 'orientation', 'mean_intensity'])

    df_measurments = pd.DataFrame(props)

    return threshold, label_image, df_measurments

def find_threshold(mask):
    return threshold_otsu(mask)

def get_watershed(label_image):
    distance = ndi.distance_transform_edt(label_image)
    local_maxi = peak_local_max(distance, footprint=np.ones((5, 5)), labels=label_image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(local_maxi.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=label_image)

    return labels

def simplify_polygon(label_image):
    polys = []
    for contour in find_contours(label_image, 0):
        coords = approximate_polygon(contour, tolerance=10)
        polys.append(coords)

    return polys

if __name__ == '__main__':

    image = io.imread(r"D:\deep_learning\data\building_footprint\test/grid_001_19_319381_270493_img.jpg")
    mask = img_as_ubyte((io.imread(r"D:\deep_learning\data\building_footprint\test_binary/grid_001_19_319381_270493_img.png")))

    threshold, label_image, df_measurments = measure(image, mask)
    watershed_polys = get_watershed(label_image)


