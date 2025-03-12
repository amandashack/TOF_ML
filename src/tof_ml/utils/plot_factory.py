# src/tof_ml/plots/plot_factory.py

import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import List, Optional, Union, Dict

logger = logging.getLogger(__name__)


class PlotFactory:
    """
    This class has stand-alone plotting methods (layered_histogram, scatter_plot, heatmap_plot, etc.).
    And a high-level plot_complex_figure method that replicates custom multi-subplot layout.
    """

    @staticmethod
    def layered_histogram(
        ax: plt.Axes,
        df: pd.DataFrame,
        variable: str,
        group_by: Optional[str] = None,
        nbins: int = 50,
        alpha: float = 0.5,
        title: Optional[str] = None,
        add_legend: bool = True,
        selected_groups: Union[int, List[float], None] = None,
        sample_n: int = 10000
    ):
        """
        Plots a layered histogram for 'variable' from df, grouping by 'group_by'.
        If 'selected_groups' is int, we do a symlog-based subset of that many groups.
        If it's a list, we only plot those group values.
        """
        if sample_n < len(df):
            df = df.sample(n=sample_n, random_state=42)

        if group_by:
            groups = df.groupby(group_by)
            group_keys = list(groups.groups.keys())

            # Sort group keys by symlog approach
            group_keys_sorted = sorted(group_keys, key=lambda x: np.sign(x) * np.log1p(abs(x)))

            # Determine which keys to plot
            if selected_groups is not None:
                if isinstance(selected_groups, int):
                    # We'll always include min, max, and possibly 0
                    desired_count = selected_groups
                    allowed = {group_keys_sorted[0], group_keys_sorted[-1]}
                    if 0 in group_keys:
                        allowed.add(0)
                    if len(allowed) < desired_count:
                        # fill from the remaining
                        remaining = [k for k in group_keys_sorted if k not in allowed]
                        additional_needed = desired_count - len(allowed)
                        if remaining:
                            indices = np.linspace(0, len(remaining) - 1, additional_needed, dtype=int)
                            for i in indices:
                                allowed.add(remaining[i])
                    allowed_keys = sorted(allowed)
                elif isinstance(selected_groups, list):
                    allowed_keys = selected_groups
                else:
                    allowed_keys = group_keys_sorted
            else:
                allowed_keys = group_keys_sorted

            n_groups = len(allowed_keys)
            cmap = plt.get_cmap("viridis", n_groups)
            color_index = 0

            for group_key in group_keys_sorted:
                if group_key not in allowed_keys:
                    continue
                group_df = groups.get_group(group_key)

                ax.hist(group_df[variable], bins=nbins,
                        color=cmap(color_index),
                        edgecolor='black',
                        alpha=alpha,
                        label=f"{group_by} = {group_key}")
                color_index += 1

            if add_legend:
                ax.legend(fontsize=8)
        else:
            ax.hist(df[variable], bins=nbins, edgecolor='black', alpha=alpha)

        ax.set_xlabel(variable)
        ax.set_ylabel('Frequency')
        ax.set_yscale('log')  # by default using log-scale on y
        if title:
            ax.set_title(title)

    @staticmethod
    def scatter_plot(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        title: str = "",
        ax: Optional[plt.Axes] = None,
        sample_n: int = 1000
    ):
        """
        Scatter plot using seaborn, with optional color by a feature,
        optionally sampling the data to avoid huge plots.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))

        # sample
        if sample_n < len(df):
            df = df.sample(n=sample_n, random_state=42)

        if color_col and color_col in df.columns:
            sns.scatterplot(
                data=df,
                x=x_col, y=y_col,
                hue=color_col,
                palette="viridis",
                ax=ax
            )
        else:
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)

    @staticmethod
    def heatmap_plot(
        df: pd.DataFrame,
        x_col: str = "mid2",
        y_col: str = "mid1",
        title: str = "",
        ax: Optional[plt.Axes] = None
    ):
        """
        Creates a pivot table of x_col vs y_col frequencies, then plots a heatmap.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))

        pivot = pd.crosstab(df[y_col], df[x_col])  # row -> y_col, col -> x_col
        if pivot.size == 0:
            logger.warning("No data for heatmap.")
            return

        vmax = pivot.values.max()
        sns.heatmap(pivot, ax=ax, cmap="viridis", vmin=0, vmax=vmax, annot=True, fmt="d")
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title if title else f"Heatmap of {y_col} vs {x_col} (Frequency)")

    @staticmethod
    def histogram_plot(
        df: pd.DataFrame,
        col_list: List[str],
        color_by: Optional[str] = None,
        title: str = "",
        out_path: str = ""
    ) -> str:
        """
        Example function that plots multiple histograms side by side.
        (You can keep or remove this if you prefer the layered approach.)
        """
        fig, axes = plt.subplots(1, len(col_list), figsize=(6*len(col_list), 4))
        if len(col_list) == 1:
            axes = [axes]  # ensure list

        for i, col in enumerate(col_list):
            ax = axes[i]
            if color_by and color_by in df.columns:
                sns.histplot(data=df, x=col, hue=color_by,
                             multiple="stack", alpha=0.7, ax=ax)
            else:
                sns.histplot(data=df, x=col, ax=ax, kde=False, color="blue")
            ax.set_xlabel(col)
        plt.suptitle(title)
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path)
            plt.close()
            return out_path
        else:
            plt.show()
            return ""

    @staticmethod
    def plot_complex_figure(
            df: pd.DataFrame,
            title: str = "",
            out_path: str = "",
            selected_groups: int = 7,
            scatter_symlog: bool = True
    ) -> str:
        import matplotlib.gridspec as gridspec

        # Choose either 'pass_energy' or 'kinetic_energy', whichever exists
        if "pass_energy" in df.columns:
            energy_var = "pass_energy"
        elif "kinetic_energy" in df.columns:
            energy_var = "kinetic_energy"
        else:
            # if neither is found, either raise an error or skip
            raise ValueError("No pass_energy or kinetic_energy column in the DataFrame!")

        fig = plt.figure(figsize=(16, 10))
        gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1, 0.4])

        # 6 axes for layered hist
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])
        ax4 = fig.add_subplot(gs[2, 0])
        ax5 = fig.add_subplot(gs[2, 1])

        # dedicated legend axis spanning the first two rows of the third column
        axLegend = fig.add_subplot(gs[:2, 2])

        # 1) layered hist: elevation by retardation
        PlotFactory.layered_histogram(
            ax0, df,
            variable='elevation',
            group_by='retardation',
            nbins=50,
            alpha=0.5,
            title=f'{title} Elevation by Retardation',
            add_legend=False,
            selected_groups=selected_groups
        )
        # 2) layered hist: pass_energy or kinetic_energy by retardation
        PlotFactory.layered_histogram(
            ax1, df,
            variable=energy_var,
            group_by='retardation',
            nbins=50,
            alpha=0.5,
            title=f'{title} {energy_var} by Retardation',
            add_legend=False,
            selected_groups=selected_groups
        )
        # 3) x by retardation
        PlotFactory.layered_histogram(
            ax2, df,
            variable='x',
            group_by='retardation',
            nbins=50,
            alpha=0.5,
            title=f'{title} X by Retardation',
            add_legend=False,
            selected_groups=selected_groups
        )
        # 4) y by retardation
        PlotFactory.layered_histogram(
            ax3, df,
            variable='y',
            group_by='retardation',
            nbins=50,
            alpha=0.5,
            title=f'{title} Y by Retardation',
            add_legend=False,
            selected_groups=selected_groups
        )

        # 5) scatter plot on ax4: (pass_energy or kinetic_energy) vs tof, color by retardation
        PlotFactory.scatter_plot(
            df,
            x_col=energy_var,
            y_col="tof",
            color_col="retardation",
            title=f"{title} Time of Flight vs {energy_var}",
            ax=ax4
        )
        if scatter_symlog:
            ax4.set_xscale('symlog', base=2)
            ax4.set_yscale('symlog', base=2)

        # 6) heatmap plot on ax5
        PlotFactory.scatter_plot(
            df,
            x_col=energy_var,
            y_col="tof",
            color_col="elevation",
            title=f"{title} Time of Flight vs {energy_var}",
            ax=ax5
        )

        if scatter_symlog:
            ax5.set_xscale('symlog', base=2)
            ax5.set_yscale('symlog', base=2)

        # Create a shared legend using the handles from one histogram axis
        handles, labels = ax0.get_legend_handles_labels()
        axLegend.legend(handles, labels, fontsize=8, loc='center')
        axLegend.axis('off')

        plt.tight_layout()
        print(out_path)
        if out_path:
            plt.savefig(out_path, dpi=150)
            plt.close()
            logger.info(f"Saved complex figure to {out_path}")
            return out_path
        else:
            print("Showing plot..")
            plt.show()
            return ""

