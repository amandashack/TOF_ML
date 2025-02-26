import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Union, List, Dict, Any
import matplotlib.pyplot as plt
import pandas as pd
import os

logger = logging.getLogger(__name__)


class PlotFactory:
    @staticmethod
    def scatter_plot(
        df: pd.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        title: str = "",
        out_path: str = "",
        ax: plt.Axes = None
    ) -> str:
        """A simple scatter plot, optional color by color_col."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6,4))

        if color_col:
            sc = ax.scatter(df[x_col], df[y_col], c=df[color_col], cmap="viridis", alpha=0.7)
            cbar = plt.colorbar(sc, ax=ax)
            cbar.set_label(color_col)
        else:
            ax.scatter(df[x_col], df[y_col], color="blue", alpha=0.7)

        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(title)

        plt.tight_layout()
        if out_path:
            plt.savefig(out_path)
            plt.close()
            return out_path
        else:
            if not ax:
                plt.show()

    @staticmethod
    def histogram_plot(
        df: pd.DataFrame,
        col_list: List[str],
        color_by: Optional[str] = None,
        title: str = "",
        out_path: str = ""
    ) -> str:
        """
        Creates one or more histograms from df, optionally stacked by 'color_by'.

        If color_by is not None, we interpret it as a categorical column
        and produce a stacked histogram using multiple="stack".
        """
        fig, axes = plt.subplots(1, len(col_list), figsize=(6*len(col_list), 4), squeeze=False)
        axes = axes.flatten()

        for i, col in enumerate(col_list):
            ax = axes[i]
            if color_by and color_by in df.columns:
                # We'll do a stacked histogram
                sns.histplot(data=df, x=col, hue=color_by,
                             multiple="stack", alpha=0.7, ax=ax, edgecolor=None)
            else:
                # Simple single histogram
                sns.histplot(data=df, x=col, ax=ax, kde=True, color="blue")

            ax.set_xlabel(col)
            ax.set_title(title if i==0 else "")

        plt.tight_layout()
        if out_path:
            plt.savefig(out_path)
            plt.close()
            return out_path
        else:
            plt.show()

    @staticmethod
    def heatmap_plot(df: pd.DataFrame, mid1_col: str = "mid1", mid2_col: str = "mid2",
                     title: str = "", out_path: str = "", ax: plt.Axes = None) -> str:
        """
        Plots a heatmap of counts for combinations of mid1 and mid2.

        This function creates a pivot table (using pd.crosstab) where the rows correspond
        to unique values of mid1 and the columns correspond to unique values of mid2. The cell
        values are the frequency (number of data points) for each combination. The heatmap's
        color scale is set from 0 to the maximum number of samples in any cell.
        """
        # Create a crosstab (pivot table) of the counts.
        pivot = pd.crosstab(df[mid1_col], df[mid2_col])
        vmax = pivot.values.max() if pivot.size > 0 else None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(pivot, ax=ax, cmap="viridis", vmin=0, vmax=vmax, annot=True, fmt="d",
                    annot_kws={"size": 8})  # smaller annotation font size

        ax.set_xlabel(mid2_col)
        ax.set_ylabel(mid1_col)
        ax.set_title(title if title else f"Heatmap of {mid1_col} vs {mid2_col} (Frequency)")
        plt.tight_layout()
        if out_path:
            plt.savefig(out_path)
            plt.close()
            return out_path
        else:
            if ax is None:
                plt.show()


class ReportGenerator:
    def __init__(self, config: Dict, plot_data_dict: Dict, output_dir: str = "./reports"):
        """
        :param config: e.g. your base_config_not_report.yaml
        :param plot_data_dict:
            {
               "data_loader": {"df": <pandas df>},
               "splitter": {"train": <df>, "val": <df>, "test": <df>},
               ...
            }
        :param output_dir: where to save plots
        """
        self.config = config
        self.plot_data_dict = plot_data_dict
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_report(self) -> Dict[str, str]:
        """
        Iterates over confirmation_plots in the config, calls the appropriate
        PlotFactory method for each. Returns {plot_name: file_path}.
        """
        plot_paths = {}
        conf_plots = self.config.get("confirmation_plots", {})

        # e.g. conf_plots might have keys: data_loader, splitter, preprocessor, trainer
        for stage_name, stage_cfg in conf_plots.items():
            stage_data = self.plot_data_dict.get(stage_name, {})
            if not stage_data:
                logger.warning(f"No data found for stage {stage_name}, skipping.")
                continue

            # stage_cfg might have "scatter", "histogram", etc.
            for plot_type, plot_instructions in stage_cfg.items():
                # e.g. plot_instructions = { "id1": {...}, "id2": {...} }
                for plot_id, plot_params in plot_instructions.items():
                    print(stage_name, stage_data, plot_type, plot_id, plot_params)
                    out_path = self._generate_plot(stage_name, stage_data, plot_type, plot_id, plot_params)
                    if out_path:
                        plot_paths[f"{stage_name}_{plot_type}_{plot_id}"] = out_path

        return plot_paths

    def _generate_plot(
        self,
        stage_name: str,
        stage_data: Dict[str, pd.DataFrame],
        plot_type: str,
        plot_id: str,
        plot_params: Dict[str, Any]
    ) -> str:
        """
        Decide which DataFrame to plot, parse config, call PlotFactory.
        """
        title = plot_params.get("title", f"{stage_name} {plot_id}")
        color_by = plot_params.get("color_by", None)

        # Are we dealing with scatter or histogram?
        if plot_type == "scatter":
            # e.g. "axes": [ "retardation", "tof" ]
            axes = plot_params.get("axes", [])
            print(axes)
            if len(axes) != 2:
                logger.warning(f"Scatter requires 2 axes. Skipping {plot_id}.")
                return ""

            # pick a df
            df = self._pick_df(stage_data, plot_params)
            print(df)
            if df is None or not all(a in df.columns for a in axes):
                logger.warning(f"Missing columns {axes} or no df. Skipping.")
                return ""

            out_file = os.path.join(self.output_dir, f"{stage_name}_{plot_type}_{plot_id}.png")
            return PlotFactory.scatter_plot(
                df=df,
                x_col=axes[0],
                y_col=axes[1],
                color_col=color_by,
                title=title,
                out_path=out_file
            )

        elif plot_type == "histogram":
            # e.g. "keys": [ "pass_energy", ... ]
            col_list = plot_params.get("keys", [])
            if isinstance(col_list, str):
                col_list = [col_list]

            df = self._pick_df(stage_data, plot_params)
            if df is None or not all(c in df.columns for c in col_list):
                logger.warning(f"Missing columns {col_list} or no df for {plot_id}.")
                return ""

            out_file = os.path.join(self.output_dir, f"{stage_name}_{plot_type}_{plot_id}.png")
            return PlotFactory.histogram_plot(
                df=df,
                col_list=col_list,
                color_by=color_by if isinstance(color_by, str) else None,
                title=title,
                out_path=out_file
            )

        else:
            logger.warning(f"Unknown plot type {plot_type} for {plot_id}.")
            return ""

    def _pick_df(self, stage_data: Dict[str, pd.DataFrame], plot_params: Dict[str, Any]) -> pd.DataFrame:
        """
        If your stage_data has multiple data frames, let the config specify which one.
        Or default to the only DataFrame if there's just one.
        """
        df_name = plot_params.get("df_name")
        if df_name and df_name in stage_data:
            return stage_data[df_name]

        if len(stage_data) == 1:
            return next(iter(stage_data.values()))

        logger.warning(f"Cannot pick a single df from stage_data for plot_params={plot_params}.")
        return None
