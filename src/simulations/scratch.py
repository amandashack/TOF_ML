import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import sys
import json
import xarray as xr
import itertools
from load_positive_voltages import DS_positive
from matplotlib.backends.backend_pdf import PdfPages
sys.path.insert(0, os.path.abspath('..'))
from loaders.load_xarrays import save_xarray, load_xarray
from utilities.plotting_tools import (plot_imagetool, plot_relation, plot_heatmap, plot_histogram,
                                      plot_energy_resolution, plot_parallel_coordinates, plot_ks_score)
from utilities.mask_data import create_mask
from utilities.calculation_tools import calculate_ks_score, normalize_3D


# decide what you want to load in
#location = r"C:\Users\proxi\Documents\coding\TOF_ML\simulations\TOF_simulation\simion_output\collection_efficiency"
#efficiency_xarray = load_xarray(location, "collection_efficiency")

def plot_collection_efficiency(xar, retardations, mid1_ratios, mid2_ratios, directory=None, filename=None):
    colors = plt.cm.viridis(np.linspace(0, 1, len(mid1_ratios)))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    for retardation in retardations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if retardation > 0:
            ke = 0.1
            title_retardation = f'+{retardation} eV Retardation'
        else:
            ke = np.abs(retardation) + 2
            title_retardation = f'-{np.abs(retardation)} eV Retardation'

        # First plot: mid1 and mid2 as x and y axes for a specific kinetic energy above the retardation
        cut = xar.sel({'kinetic_energy': ke, 'retardation': retardation})
        cut.plot(x='mid1_ratio', y='mid2_ratio', ax=ax1)

        ax1.set_title(f'Collection Efficiency for {title_retardation} and {ke} eV KE')
        ax1.set_xlabel('mid1_ratio')
        ax1.set_ylabel('mid2_ratio')
        ax1.grid(True)

        # Second plot: Kinetic energy on x axis with mid1 as the same color but different linestyle for each mid2
        for i, mid1_ratio in enumerate(mid1_ratios):
            for j, mid2_ratio in enumerate(mid2_ratios):
                cut = xar.sel({'retardation': retardation, 'mid1_ratio': mid1_ratio, 'mid2_ratio': mid2_ratio})
                cut.plot(ax=ax2, color=colors[i], linestyle=linestyles[j % len(linestyles)])

        ax2.set_title(f'Collection Efficiency vs. Kinetic Energy for {title_retardation}')
        ax2.set_xlabel('Kinetic Energy (eV)')
        ax2.set_ylabel('Collection Efficiency')
        ax2.grid(True)

        # Create custom legend entries
        handles1 = [plt.Line2D([0], [0], color=colors[i], lw=4, label=f'mid1: {mid1_ratio}') for i, mid1_ratio in enumerate(mid1_ratios)]
        handles2 = [plt.Line2D([0], [0], color='black', linestyle=linestyles[j % len(linestyles)], label=f'mid2: {mid2_ratio}') for j, mid2_ratio in enumerate(mid2_ratios)]

        # Adding legends
        ax2.legend(handles=handles1 + handles2, loc='best')

        if directory and filename:
            pdf_pages.savefig(fig)
            plt.close()
        else:
            plt.show()

    if directory and filename:
        pdf_pages.close()


def plot_xar_instances(xar, retardations, mid1_ratios, mid2_ratios, value_def="Collection Efficiency", directory=None, filename=None):
    colors = plt.cm.viridis(np.linspace(0, 1, len(mid1_ratios)))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    for retardation in retardations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if retardation > 0:
            ke = 0.1
            title_retardation = f'+{retardation} eV Retardation'
        else:
            ke = np.abs(retardation) + 2
            title_retardation = f'-{np.abs(retardation)} eV Retardation'

        # First plot: mid1 and mid2 as x and y axes for a specific kinetic energy above the retardation
        cut = xar.sel({'kinetic_energy': ke, 'retardation': retardation})
        cut.plot(x='mid1_ratio', y='mid2_ratio', ax=ax1, cmap='bwr')

        ax1.set_title(f'{value_def} for {title_retardation} and {ke} eV KE')
        ax1.set_xlabel('Blade 22')
        ax1.set_ylabel('Blade 25')
        ax1.grid(True)

        # Second plot: Kinetic energy on x axis with mid1 as the same color but different linestyle for each mid2
        for i, mid1_ratio in enumerate(mid1_ratios):
            for j, mid2_ratio in enumerate(mid2_ratios):
                cut = xar.sel({'retardation': retardation, 'mid1_ratio': mid1_ratio, 'mid2_ratio': mid2_ratio})
                cut.plot(ax=ax2, color=colors[i], linestyle=linestyles[j % len(linestyles)])

        ax2.set_title(f'{value_def} vs. Kinetic Energy for {title_retardation}')
        ax2.set_xlabel('Kinetic Energy (eV)')
        ax2.set_ylabel(f'{value_def}')
        ax2.grid(True)

        # Create custom legend entries
        handles1 = [plt.Line2D([0], [0], color=colors[i], lw=4, label=f'Blade 22: {mid1_ratio}')
                    for i, mid1_ratio in enumerate(mid1_ratios)]
        handles2 = [plt.Line2D([0], [0], color='black', linestyle=linestyles[j % len(linestyles)],
                               label=f'Blade 25: {mid2_ratio}') for j, mid2_ratio in enumerate(mid2_ratios)]

        # Adding legends
        ax2.legend(handles=handles1 + handles2, loc='best')

        if directory and filename:
            pdf_pages.savefig(fig)
            plt.close()
        else:
            plt.show()

    if directory and filename:
        pdf_pages.close()

def main(xar, mid1_ratios, mid2_ratios):
    fig, ax = plt.subplots()
    what_you_want = {'retardation': [5, 4, 2, 1, 0, -1, -2, -5]}

    colors = plt.cm.viridis(np.linspace(0, 1, len(what_you_want['retardation'])))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    for i, retardation in enumerate(what_you_want['retardation']):
        color = colors[i]
        for j, (mid1_ratio, mid2_ratio) in enumerate(itertools.product(mid1_ratios, mid2_ratios)):
            linestyle = linestyles[j % len(linestyles)]
            cut = xar.sel({'retardation': retardation, 'mid1_ratio': mid1_ratio, 'mid2_ratio': mid2_ratio})
            cut.plot(ax=ax, color=color, linestyle=linestyle,
                     label=f'Retardation: {retardation} eV, mid1: {mid1_ratio}, mid2: {mid2_ratio}')

    # Create custom legend entries
    handles = []

    # Colors for retardations
    for i, retardation in enumerate(what_you_want['retardation']):
        handles.append(plt.Line2D([0], [0], color=colors[i], lw=4, label=f'Retardation: {retardation} eV'))

    # Linestyles for mid1_ratio and mid2_ratio combinations
    for j, (mid1_ratio, mid2_ratio) in enumerate(itertools.product(mid1_ratios, mid2_ratios)):
        handles.append(plt.Line2D([0], [0], color='black', linestyle=linestyles[j % len(linestyles)],
                                  label=f'mid1: {mid1_ratio}, mid2: {mid2_ratio}'))

    ax.set_title('Collection Efficiency vs. Kinetic Energy')
    ax.set_xlabel('Kinetic Energy (eV)')
    ax.set_ylabel('Collection Efficiency')
    ax.grid(True)

    ax.legend(handles=handles, loc='best')

    plt.show()


path = r"C:\Users\proxi\Documents\coding\TOF_data"
ce_xar = load_xarray(path, "avg_tof")
retardations = [10, 5, 3, 1, 0, -1, -3, -5, -10]
mid1_ratios = [0.08, 0.11248, 0.2, 0.8]  # Example list of mid1_ratio
mid2_ratios = [0.1354, 0.3, 0.4]  # Example list of mid2_ratio
d = r"C:\Users\proxi\Documents\coding\TOF_ML\figures\shack"
#plot_xar_instances(ce_xar, retardations, mid1_ratios, mid2_ratios,
#                   directory=d, filename="collection_efficiency_comb")
plot_imagetool(ce_xar.sel({'kinetic_energy': 0.1}))

#data_loader = DS_positive()
#data_loader.load_data('simulation_data.json', xtof_range=(403.6, np.inf), ytof_range=(-13.74, 13.74),
#                      retardation_range=(-10, 10), mid1_range=(0.11248, 0.11248), mid2_range=(0.1354, 0.1354),
#                      overwrite=False)
#location = r"C:\Users\proxi\Documents\coding\TOF_ML\figures\shack"
#avg_ks_scores = plot_ks_score(data_loader.data_masked, bootstrap=10, directory=location, filename="NM_ks_scores.pdf")

# decide what you want to plot
#filtered_data = [entry for entry in data_loader.data_masked if entry['retardation'] == 1
#                 and entry['kinetic_energy'] == 20]
#fig, ax = plt.subplots()
#ax.hist(np.log(filtered_data[0]['tof_values']), bins=50, edgecolor='black', alpha=0.5)
#plt.show()
