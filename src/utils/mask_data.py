import numpy as np


def create_mask(data, x_tof_range, y_tof_range):
    xtof = np.asarray(data["x_tof"])[:].astype(float)
    ytof = np.abs(np.asarray(data["y_tof"])[:].astype(float))
    xmin_mask = xtof > x_tof_range[0]
    xmax_mask = xtof < x_tof_range[1]
    ymin_mask = ytof > y_tof_range[0]
    ymax_mask = ytof < y_tof_range[1]
    mask = xmin_mask & xmax_mask & ymin_mask & ymax_mask
    return mask