import numpy as np
from scipy.ndimage import maximum_filter, binary_fill_holes, distance_transform_edt, label
from skimage import io, morphology, filters, exposure, color, transform, measure, feature


# Colour deconvolution matrix
colour_deconv_matrix = np.array([
    [.26451728, .5205347, .81183386],
    [.9199094, .29797825, .25489032],
    [.28947765, .80015373, .5253158]
])


def process_4x(image_filename):
    """
    Process widefield images, returning a coloured image
    containing segmentations of inflammatory areas and veins.
    """

    # Read in the image
    image = transform.rescale(io.imread(image_filename), 0.25)

    # Deconvolve the image
    heu = color.separate_stains(
        image,
        np.linalg.inv(colour_deconv_matrix)
    ).astype(float)


    # INFLAMMATION

    # Apply CLAHE
    equalised_hist = exposure.equalize_adapthist(
        exposure.rescale_intensity(heu[:, :, 1], out_range=(0, 1)),
        ntiles_y=1
    )

    # Blur
    blurred = filters.gaussian(equalised_hist, 5)

    # Sigmoid transform for contrast
    contrast = exposure.adjust_sigmoid(blurred, cutoff=0.6)

    # Take an adaptive threshold
    thresholded = filters.threshold_adaptive(
        contrast,
        75,
        offset=-0.12
    )

    # Remove small connected components
    cleaned = morphology.remove_small_objects(thresholded, 250)

    # Enlarge areas
    enlarged = maximum_filter(cleaned, 11)

    # Binary closing
    inflammation = morphology.closing(enlarged, morphology.disk(15))


    # VEINS

    # How stained by eosin is each individual pixel ?
    unstained = heu[:, :, 2] / heu[:, :, 0]

    # Blur
    blurred = filters.gaussian(unstained, 11)

    # Thresholded
    thresholded = filters.threshold_adaptive(blurred, 251, offset=-0.13)

    # Morphlogical closing
    closed = morphology.binary_closing(thresholded, morphology.disk(25))

    # Remove small connected components
    veins = morphology.remove_small_objects(closed, 200)


    # Inflammation in blue, veins in green
    coloured = np.zeros_like(heu)
    coloured[:, :, 1] = veins
    coloured[:, :, 2] = inflammation

    return coloured


def process_10x(image_filename):
    """
    Process narrowfield images, returning the segmentation of the nuclei
    as well as the contour of the central inflammatory zone.
    """

    # NUCLEI

    # Read in the image at full resolution
    image = io.imread(image_filename)

    # Increase contrast
    contrast = exposure.adjust_sigmoid(image[:, :, 0], cutoff=0.6, gain=20)

    # Thresholded
    thresholded = filters.threshold_adaptive(contrast, 151)

    closed = morphology.binary_closing(thresholded, morphology.disk(3))

    # Remove small connected components
    nuclei = -morphology.remove_small_objects(closed, 100)


    # CONTOUR OF CENTRAL INFLAMMATORY ZONE

    # Local nuclear density
    local_density = exposure.rescale_intensity(
        filters.gaussian(nuclei, 61),
        out_range=(0, 1)
    )

    # Local entropy of local_density, smoothed
    entropy = exposure.rescale_intensity(
        filters.gaussian(
            filters.rank.entropy(local_density, morphology.disk(3)),
            75
        ),
        out_range=(0, 1)
    )

    # Local "information", or structure
    info = entropy * (1 + local_density)

    # Remove boundary effects
    mask = np.zeros_like(info)
    mask[200:-200, 200:-200] = 1
    info *= mask

    # Threshold the local information
    info_thresholded = (info) > filters.threshold_otsu(info)

    # Find contours
    contours = measure.find_contours(info_thresholded, 0.5)

    # Identify the centroids of all contours
    centroids = []
    vals = []
    for contour in contours :
        mean_x = contour[:, 1].mean() - info_thresholded.shape[1] / 2
        mean_y = contour[:, 0].mean() - info_thresholded.shape[0] / 2
        centroids.append(np.linalg.norm([mean_x, mean_y]))
        vals.append(local_density.T[contour.astype(int)].sum())

    # Find the largest, most central contour
    central_contour = np.fliplr(contours[np.argmin(centroids / np.array(vals))])


    # NUCLEAR SEGMENTATION

    # Maximum distance between nuclear boundaries
    distances = distance_transform_edt(nuclei)

    # Peaks of each nucleus
    peaks = feature.peak_local_max(distances, indices=False, labels=nuclei)

    # Label each peak with its own value
    labelled_peaks = label(peaks)[0]

    # Segment using the watershed algorithm
    segmented_nuclei = morphology.watershed(-distances, labelled_peaks, mask=nuclei)


    # Return the segmented image and the central contour
    return segmented_nuclei, central_contour
