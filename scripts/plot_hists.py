import random
from src.tof_ml.data.h5_data_loader import H5DataLoader
from src.tof_ml.data.column_mapping import COLUMN_MAPPING, REDUCED_COLUMN_MAPPING
from src.tof_ml.database.report_generator import PlotFactory
from src.tof_ml.utils.plotting_tools import plot_histogram, plot_layered_histogram
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def group_builder():
    sign = None # Set this
    v1 = 0
    v2 = 0
    v3 = 0
    v4 = 0
    v5 = 0
    v6 = 0
    v7 = 0
    v8 = 0
    v9 = 0
    v10 = 0
    v11 = 0
    v12 = 0
    v13 = 0
    v14 = 0
    v15 = 0
    v16 = 0
    v17 = 0
    v18 = 0
    v19 = 0
    v20 = 0
    v21 = 0
    v22 = 1
    v23 = 0
    v24 = 0
    v25 = 1
    R = 1
    PE = 1


def plot_hists(data, title):
    import matplotlib.gridspec as gridspec

    # Create a grid layout that reserves a dedicated column for a shared legend.
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 0.4])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 0])
    ax5 = fig.add_subplot(gs[2, 1])
    # Dedicated legend axis spanning the first two rows of the third column.
    axLegend = fig.add_subplot(gs[:2, 2])

    # Here we specify selected_groups explicitly as a list:
    plot_layered_histogram(ax0, data, variable='elevation', x_label='Initial Elevation',
                           title=f'{title} Elevation by Retardation', nbins=50, group_by='retardation',
                           alpha=0.5, add_legend=False, selected_groups=7)
    # Plot other histograms similarly.
    plot_layered_histogram(ax1, data, variable='pass_energy', x_label='Pass Energy',
                           title=f'{title} Pass Energy by Retardation', nbins=50, group_by='retardation',
                           alpha=0.5, add_legend=False, selected_groups=7)
    plot_layered_histogram(ax2, data, variable='x', x_label='Final X Position',
                           title=f'{title} X Position by Retardation', nbins=50, group_by='retardation',
                           alpha=0.5, add_legend=False, selected_groups=7)
    plot_layered_histogram(ax3, data, variable='y', x_label='Final Y Position',
                           title=f'{title} Y Position by Retardation', nbins=50, group_by='retardation',
                           alpha=0.5, add_legend=False, selected_groups=7)

    data_filtered = data.sample(n=5_000)

    PlotFactory.scatter_plot(data_filtered, x_col="pass_energy", y_col="tof",
                             color_col="retardation", title=f"{title} Time of Flight vs Pass Energy", ax=ax4)
    PlotFactory.heatmap_plot(data, ax=ax5)

    # Set log scales on the scatter plot axes.
    ax4.set_xscale('symlog', base=2)
    ax4.set_yscale('symlog', base=2)

    # Now create a shared legend using the handles from one of the histogram axes.
    handles, labels = ax0.get_legend_handles_labels()
    axLegend.legend(handles, labels, fontsize=8, loc='center')
    axLegend.axis('off')  # Hide the axis for the legend.

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from src.tof_ml.utils.config_utils import load_config
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from src.tof_ml.logging.logging_utils import setup_logger

    logger = setup_logger('trainer')

    base_config = load_config("config/base_config.yaml")
    data_config = base_config.get("data", {})
    data_loader = H5DataLoader(config=data_config, column_mapping=COLUMN_MAPPING)
    # Load data
    df = data_loader.load_data()

    df_masked = pd.DataFrame(df, columns=data_loader.column_mapping.keys())

    df_raw = data_loader.raw_data
    df_raw_filtered = pd.DataFrame(df_raw, columns=REDUCED_COLUMN_MAPPING.keys())

    plot_hists(df_raw_filtered, "Raw Data")
    plot_hists(df_masked, "Masked Data")

    # print statistics
    # What percentage of points remain after masking for each combination of mid1, mid2, retardation?
    # Let's start with a random point.
    def print_row(target):
        for i, column in enumerate(REDUCED_COLUMN_MAPPING.keys()):
            print(column, target[i])
        print('--- End of row ---\n')


    """# Create a generator for the rows (assuming df_raw is indexable)
    row_gen = (df_raw[i, :].flatten() for i in range(df_raw.shape[0]))

    # Print the first 6 rows
    for _ in range(6):
        try:
            row = next(row_gen)
        except StopIteration:
            break
        print_row(row)"""

    random_indices = random.sample(range(df_raw.shape[0]), 6)

    # Print the 6 randomly selected rows
    for idx in random_indices:
        # Assuming df_raw is indexable as df_raw[i, :].flatten()
        row = df_raw[idx, :].flatten()
        print_row(row)

    #PlotFactory.histogram_plot(df_raw_filtered, col_list=["x"], color_by="retardation")
    #PlotFactory.histogram_plot(df_raw_filtered, col_list=["elevation"], color_by="kinetic_energy")
    #PlotFactory.histogram_plot(df_masked, col_list=["elevation"], color_by="kinetic_energy")

    # Create a scatter plot using seaborn where:
    # - x = "time_of_flight"
    # - y = "retardation"
    # - hue = "kinetic_energy" (the point color scale)
    plt.figure(figsize=(8, 6))
    scatter_plot = sns.scatterplot(
        data=df_masked.sample(n=5_000),
        x="pass_energy",
        y="retardation",
        hue="elevation",
        palette="viridis"
    )

    # Use a log (base-2) scale on both axes
    # Note: You can use plt.xscale('log', base=2) in newer matplotlib,
    # or plt.xscale('log', basex=2) for older versions.
    plt.xscale('symlog', base=2)
    plt.yscale('symlog', base=2)

    # Label the axes and show the plot
    plt.xlabel("Pass Energy (log2 scale)")
    plt.ylabel("Retardation (log2 scale)")
    plt.title("Masked Retardation vs Pass Energy colored by Elevation")

    plt.tight_layout()
    plt.show()
