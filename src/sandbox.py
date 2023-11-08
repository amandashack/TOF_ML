"""
This file is a sandbox for testing/running functions
"""

from loaders import MRCOLoader
from plotter import one_plot_multi_scatter
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import arpys
from model_gen import run_model


def multi_scatter(spec):
    fig, ax = plt.subplots()
    ax = one_plot_multi_scatter(ax, spec, "multi retardation", "Log2(pass energy)", "Log2(time of flight)")
    fig.tight_layout()
    plt.legend(prop={'size': 6})
    plt.show()


if __name__ == '__main__':
    amanda_filepath = "C:/Users/proxi/Downloads/NM_simulations"
    multi_retardation_sim = MRCOLoader(amanda_filepath)
    multi_retardation_sim.load()
    multi_retardation_sim.create_mask((402, np.inf), (0, 13.7), "make it")
    #multi_scatter(multi_retardation_sim)
    run_model(multi_retardation_sim.data_masked)