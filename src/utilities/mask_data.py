import numpy as np


def create_mask(data, x_tof_range, y_tof_range=None):
    """
    Create a mask based on x_tof and y_tof ranges.

    Parameters:
    - data (dict): Dictionary containing 'x_tof' and 'y_tof' arrays.
    - x_tof_range (tuple): (min_x_tof, max_x_tof)
    - y_tof_range (tuple, optional): (min_y_tof, max_y_tof)

    Returns:
    - np.ndarray: Boolean mask array.
    """
    xtof = np.asarray(data["x_tof"]).astype(float)
    ytof = np.abs(np.asarray(data["y_tof"]).astype(float))

    # Create masks for x_tof range
    xmin_mask = xtof > x_tof_range[0]
    xmax_mask = xtof < x_tof_range[1]
    mask = xmin_mask & xmax_mask

    # If y_tof_range is provided, include it in the mask
    if y_tof_range is not None:
        ymin_mask = ytof > y_tof_range[0]
        ymax_mask = ytof < y_tof_range[1]
        mask &= ymin_mask & ymax_mask

    return mask