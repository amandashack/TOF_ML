import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import xarray as xr
import tensorflow as tf


class LRFinder(tf.keras.callbacks.Callback):
    def __init__(self, min_lr=1e-8, max_lr=10, steps_per_epoch=None, epochs=1):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_steps = steps_per_epoch * epochs
        self.step = 0
        self.lr_mult = (max_lr / min_lr) ** (1 / self.total_steps)
        self.losses = []
        self.lrs = []

    def on_batch_end(self, batch, logs=None):
        self.step += 1
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        lr *= self.lr_mult
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)
        self.lrs.append(lr)
        self.losses.append(np.log(logs['loss']))

        # Debugging: Print the learning rate and loss
        print(f"Step: {self.step}, Learning Rate: {lr}, Loss: {logs['loss']}")

        if self.step >= self.total_steps:
            self.model.stop_training = True

    def plot_loss(self, sma=1):
        # Apply simple moving average (SMA) to smooth the loss plot
        def sma_smoothing(values, window):
            weights = np.repeat(1.0, window) / window
            return np.convolve(values, weights, 'valid')

        smoothed_losses = sma_smoothing(self.losses, sma)

        plt.plot(self.lrs[:len(smoothed_losses)], smoothed_losses)
        plt.xscale('log')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate Finder')
        plt.show()

def calculate_ks_score(y_pos, num_bootstrap=10, R=13.74, plot=False, print_ks=False):
    ks_scores = []
    num_points = len(y_pos)
    if num_points < 2:
        return np.nan

    uniform_points = []
    final_points = []

    for _ in range(num_bootstrap):
        theta_uniform = np.random.uniform(0, 2 * np.pi, num_points)
        radius_uniform = np.random.uniform(0, R**2, num_points) ** 0.5
        x_uniform = radius_uniform * np.cos(theta_uniform)
        y_uniform = radius_uniform * np.sin(theta_uniform)

        theta_final = np.random.uniform(0, 2 * np.pi, num_points)
        radius_final = np.abs(y_pos**2) ** 0.5
        x_final = radius_final * np.cos(theta_final)
        y_final = radius_final * np.sin(theta_final)

        uniform_points.append((x_uniform, y_uniform))
        final_points.append((x_final, y_final))

        uniform_dist = np.hstack((x_uniform, y_uniform))
        final_dist = np.hstack((x_final, y_final))

        if len(uniform_dist) == 0 or len(final_dist) == 0:
            continue

        ks_score, _ = ks_2samp(final_dist, uniform_dist)
        if print_ks:
            print(ks_score)
        ks_scores.append(ks_score)

    if plot:
        colors = plt.cm.viridis(np.linspace(0, 1, num_bootstrap))

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        for i in range(num_bootstrap):
            ax[0].scatter(uniform_points[i][0], uniform_points[i][1], alpha=0.5, color=colors[i])
            ax[1].scatter(final_points[i][0], final_points[i][1], alpha=0.5, color=colors[i])

        ax[0].set_title('Uniform Distribution')
        ax[0].set_xlabel('X')
        ax[0].set_ylabel('Y')
        ax[0].set_aspect('equal', 'box')

        ax[1].set_title('Final Elevation Distribution')
        ax[1].set_xlabel('X')
        ax[1].set_ylabel('Y')
        ax[1].set_aspect('equal', 'box')

        plt.tight_layout()
        plt.show()

    return np.mean(ks_scores) if ks_scores else np.nan

def standardize(ar):
    """
    Makes the spectra intensities between 0 and 1
    :param ar: xarray with data
    :return: xarray standardized to between 0 and 1
    """
    w_max = float(np.nanmax(ar))
    w_min = float(np.nanmin(ar))
    nr_values = (ar - w_min) / (w_max - w_min)
    return nr_values


def normalize_2D(*args):
    spectra = []
    for arg in args:
        dims = arg.dims
        energy = arg[dims[1]]
        kx = arg[dims[0]]
        sum_thing = np.nansum(arg.values)
        area = arg[dims[1]].values.size * arg[dims[0]].values.size
        average = sum_thing / area
        cut_normed = arg / average
        st = standardize(cut_normed)
        spectra.append(xr.DataArray(st, coords={dims[0]: kx, dims[1]: energy},
                                    dims = [dims[0], dims[1]], attrs=arg.attrs))
    return spectra


def normalize_3D(arg):
    dims = arg.dims
    ax2 = arg[dims[2]].values
    ax0 = arg[dims[0]].values
    ax1 = arg[dims[1]].values
    sum_thing = np.nansum(arg.values)
    volume = len(ax0) * len(ax1) * len(ax2)
    average = sum_thing / volume
    cut_normed = arg / average
    st = standardize(cut_normed)
    return xr.DataArray(st, coords={dims[0]: ax0, dims[1]: ax1, dims[2]: ax2},
                                     dims=[dims[0], dims[1], dims[2]], attrs=arg.attrs)