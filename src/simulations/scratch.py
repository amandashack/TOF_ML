import numpy as np
import os
import sys
import itertools
from load_positive_voltages import DS_positive
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
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

def plot_xar_single_axis(xar, retardations, ratios, value_def="Collection Efficiency", directory=None, filename=None):
    colors = plt.cm.viridis(np.linspace(0, 1, len(retardations)))
    linestyles = ['-', '--', ':']
    lineweights = np.linspace(2, 3, len(retardations))  # Adjust line weights from 1 to 3

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, retardation in enumerate(retardations):
        for j, (mid1_ratio, mid2_ratio) in enumerate(ratios):
            cut = xar.sel({'retardation': retardation, 'mid1_ratio': mid1_ratio, 'mid2_ratio': mid2_ratio})
            cut.plot(ax=ax, color=colors[i], linestyle=linestyles[j % len(linestyles)],
                     linewidth=lineweights[i], label=f'Ret: {retardation}\nM1: {mid1_ratio}, M2: {mid2_ratio}')

    ax.set_title(f'{value_def} vs. Kinetic Energy')
    ax.set_xlabel('Kinetic Energy (eV)')
    ax.set_ylabel(f'{value_def}')
    ax.set_ylim([0, 1])
    ax.grid(True)

    # Create custom legend entries
    handles = [plt.Line2D([0], [0], color=colors[i], linestyle=linestyles[j % len(linestyles)], linewidth=lineweights[i],
                          label=f'Ret: {retardation}\nM1: {mid1_ratio}, M2: {mid2_ratio}')
               for i, retardation in enumerate(retardations)
               for j, (mid1_ratio, mid2_ratio) in enumerate(ratios)]

    # Adding legends outside the plot
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='medium')

    plt.tight_layout()

    if directory and filename:
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close()
        pdf_pages.close()
    else:
        plt.show()


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


def plot_collection_efficiency_grid(xar, retardations, kinetic_energies, directory=None, filename=None):
    num_plots = len(retardations)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols  # Calculate number of rows needed

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 5))
    axes = axes.flatten()

    for idx, (retardation, ke) in enumerate(zip(retardations, kinetic_energies)):
        ax = axes[idx]
        cut = xar.sel({'kinetic_energy': ke, 'retardation': retardation})
        im = cut.plot(x='mid1_ratio', y='mid2_ratio', ax=ax, vmin=0, vmax=1, cmap='bwr')

        if retardation > 0:
            title_retardation = f'R = +{retardation} V, KE = {ke} eV'
        else:
            title_retardation = f'R = {retardation} V, KE = {ke} eV'

        ax.set_title(title_retardation)
        ax.set_xlabel('Blade 22')
        ax.set_ylabel('Blade 25')
        ax.grid(True)

    # Hide any remaining empty subplots
    for idx in range(len(retardations), len(axes)):
        fig.delaxes(axes[idx])

    plt.tight_layout()

    if directory and filename:
        pdf_pages.savefig(fig, bbox_inches='tight')
        plt.close()
        pdf_pages.close()
    else:
        plt.show()


def plot_xar_instances(xar, retardations, mid1_ratios, mid2_ratios, value_def="Collection Efficiency", directory=None, filename=None):
    colors = plt.cm.viridis(np.linspace(0, 1, len(mid1_ratios)))
    linestyles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 1))]

    if directory and filename:
        output_path = os.path.join(directory, f'{filename}.pdf')
        pdf_pages = PdfPages(output_path)

    for retardation in retardations:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        if retardation > 0:
            ke = 10
            title_retardation = f'+{retardation} eV Retardation'
        else:
            ke = np.abs(retardation) + 10
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
ce_xar = load_xarray(path, "collection_efficiency")
retardations = [-1, -10]
mid1_ratios = [0.08, 0.11248, 0.2, 0.8]  # Example list of mid1_ratio
mid2_ratios = [0.1354, 0.3, 0.4]  # Example list of mid2_ratio
ratios = [(0.11248, 0.1354), (0.2, 0.3), (0.8, 0.7)]
d = r"C:\Users\proxi\Documents\coding\TOF_ML\figures\shack"
#plot_xar_instances(ce_xar, retardations, mid1_ratios, mid2_ratios)
#plot_xar_single_axis(ce_xar, retardations, ratios,
#                   directory=d, filename="collection_efficiency_neg", value_def="Collection Efficiency")
#retardations = [10, 1, 10, 1]
#kinetic_energies = [0.1, 3, 10, 11]
#plot_collection_efficiency_grid(ce_xar, retardations, kinetic_energies,
#                                directory=d, filename="ncollections_pos")
plot_imagetool(ce_xar.sel({'retardation': 10}))
print(ce_xar.sel({'retardation': 1, 'mid1_ratio': 0.3, 'mid2_ratio': 0.2, 'kinetic_energy': 0.1}),
      ce_xar.sel({'retardation': 1, 'mid1_ratio': 0.3, 'mid2_ratio': 0.2, 'kinetic_energy': 4}),
      ce_xar.sel({'retardation': 1, 'mid1_ratio': 0.3, 'mid2_ratio': 0.2, 'kinetic_energy': 9}),
      ce_xar.sel({'retardation': 1, 'mid1_ratio': 0.3, 'mid2_ratio': 0.2, 'kinetic_energy': 12}),
      ce_xar.sel({'retardation': 1, 'mid1_ratio': 0.3, 'mid2_ratio': 0.2, 'kinetic_energy': 18}))

#data_loader = DS_positive()
#data_loader.load_data('simulation_data.json', xtof_range=(403.6, np.inf), ytof_range=(-13.74, 13.74),
#                      retardation_range=(-10, 10), overwrite=False, mid1_range=(0.11248, 0.11248), mid2_range=(0.1354, 0.1354))
#avg_ks_scores = plot_ks_score(data_loader.data_masked, retardations, ratios[2][0], ratios[2][1],
#                              bootstrap=10, directory=d, filename="ks_scores_neg10_8_7.pdf",
#                              kinetic_energies=[0.1, 4, 9, 12, 18])

# decide what you want to plot
#filtered_data = [entry for entry in data_loader.data_masked if entry['retardation'] == 1
#                 and entry['kinetic_energy'] == 20]
#fig, ax = plt.subplots()
#plot_relation(ax, data_loader.data_masked, 'initial_ke', 'tof_values', 'Kinetic Energy', 'Time of Flight', title=None,
#                  plot_log=True, retardation=None, kinetic_energy=None, mid1_ratio=None, mid2_ratio=None,
#                  collection_efficiency=None, ks_score=None, verbose=False)
#output_path = os.path.join(d, 'tof_scaling.pdf')
#pdf_pages = PdfPages(output_path)
#pdf_pages.savefig(fig)
#plt.close()
#pdf_pages.close()

#ax.hist(np.log(filtered_data[0]['tof_values']), bins=50, edgecolor='black', alpha=0.5)
#plt.show()
