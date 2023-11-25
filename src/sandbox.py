"""
This file is a sandbox for testing/running functions
"""

from loaders import MRCOLoader
from plotter import one_plot_multi_scatter, pass_versus_counts
import matplotlib.pyplot as plt
import numpy as np
from model_gen import run_model
import sys


def multi_scatter(spec):
    fig, ax = plt.subplots()
    ax = one_plot_multi_scatter(ax, spec, "multi retardation", "Log2(pass energy)", "Log2(time of flight)")
    fig.tight_layout()
    plt.legend(prop={'size': 6})
    plt.show()


def run_sandbox(epochs):
    #amanda_filepath = "C:/Users/proxi/Downloads/NM_simulations"
    amanda_filepath = "/home/ajshack/TOF_ML/src/NM_simulations"
    multi_retardation_sim = MRCOLoader(amanda_filepath)
    multi_retardation_sim.load()
    multi_retardation_sim.create_mask((402, np.inf), (0, 17.7), "make it")
    # pass_versus_counts(multi_retardation_sim.spec_masked,
    #                    [multi_retardation_sim.retardation[0], multi_retardation_sim.retardation[-1]])
    # multi_scatter(multi_retardation_sim.spec_masked)
    run_model(multi_retardation_sim.data_masked, epochs=int(epochs))


if __name__ == '__main__':
    epochs = sys.argv[1]
    run_sandbox(epochs)
