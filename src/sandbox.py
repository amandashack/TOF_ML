"""
This file is a sandbox for testing/running functions
"""

from loaders import multi_retardation_loader, MRCOLoader
from plotter import one_plot_multi_scatter
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import arpys
from PyQt5.QtWidgets import QApplication, QWidget
from pyimagetool import imagetool


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
    multi_retardation_sim.organize_data()
    multi_scatter(multi_retardation_sim.spec_dict)