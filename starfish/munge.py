import numpy as np
import pandas as pd


def swap(img):
    img_swap = img.swapaxes(0, img.ndim - 1)
    return img_swap


def stack_to_list(stack):
    num_ims = stack.shape[0]
    return [stack[im, :] for im in range(num_ims)]


def list_to_stack(list):
    return np.array(list)


def max_proj(stack):
    im = np.max(stack, axis=0)
    return im


def scale(stack, metric, clip=False):
    from starfish.stats import stack_describe
    stats = stack_describe(stack)
    ims = stack_to_list(stack)
    res = [im / s[metric] for im, s in zip(ims, stats)]
    return list_to_stack(res)


def gather(df, key, value, cols):
    id_vars = [ col for col in df.columns if col not in cols ]
    id_values = cols
    var_name = key
    value_name = value
    return pd.melt( df, id_vars, id_values, var_name, value_name )


def relabel(image):
    '''
    This is a local implementation of centrosome.cpmorphology.relabel
    to remove this dependency from starfish.

    Takes a labelled image and relabels each image object consecutively.
    Original code from:
    https://github.com/CellProfiler/centrosome/blob/master/centrosome/cpmorphology.py

    They use a BSD-3 license, which would then have to be propagated to starfish,
    this could be an issue.

    Args
    ----
    image: numpy.ndarray
        A 2d integer array representation of an image with labels

    Returns
    -------
    new_image: numpy.ndarray
        A 2d integer array representation of an image wiht new labels

    n_labels: int
        The number of new unique labels
    '''

    # I've set this as a separate function, rather than binding it to the
    # WatershedSegmenter object for now

    unique_labels = set(image[image != 0])
    n_labels = len(unique_labels)

    # if the image is unlabelled, return original image
    # warning/message required?
    if n_labels == 0:
        return (image, 0)

    consec_labels = np.arange(n_labels) + 1
    lab_table = np.zeros(max(unique_labels) + 1, int)
    lab_table[[x for x in unique_labels]] = consec_labels

    # Use the label table to remap all of the labels
    new_image = lab_table[image]

    return new_image, n_labels