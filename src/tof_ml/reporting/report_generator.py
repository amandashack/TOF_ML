#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Report Generator for the ML Provenance Tracker Framework.
This module generates reports and visualizations of results.
"""

import os
import logging
import json
import datetime
import importlib
from typing import Dict, Any, Optional, List, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure

from src.tof_ml.data.data_manager import DataManager

logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    Generates reports, visualizations, and summaries of ML pipeline results.
    
    This class is responsible for:
    - Creating standard visualizations of data distributions
    - Plotting model performance metrics
    - Creating data quality reports
    - Generating model evaluation reports
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_manager: Optional[DataManager] = None,
        training_results: Optional[Dict[str, Any]] = None,
        evaluation_results: Optional[Dict[str, Any]] = None,
        output_dir: str = "./reports"
    ):
        """
        Initialize the report generator.
        
        Args:
            config: Configuration dictionary
            data_manager: Optional data manager instance
            training_results: Optional training results
            evaluation_results: Optional evaluation results
            output_dir: Directory to save reports
        """
        self.config = config
        self.data_manager = data_manager
        self.training_results = training_results or {}
        self.evaluation_results = evaluation_results or {}
        self.output_dir = output_dir
        
        # Report configuration
        self.report_config = config.get("reporting", {})
        
        # Initialize plotting style
        self._setup_plotting_style()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info("ReportGenerator initialized")
    
    def _setup_plotting_style(self):
        """Set up matplotlib plotting style."""
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
    
    def generate_reports(self) -> Dict[str, str]:
        """
        Generate all reports based on configuration.
        
        Returns:
            Dictionary of report paths
        """
        logger.info("Generating reports")
        
        report_paths = {}
        
        # Check which reports are enabled
        enabled_reports = self.report_config.get("enabled_reports", ["data", "training", "evaluation"])
        
        # Generate data distribution report if enabled
        if "data" in enabled_reports and self.data_manager:
            data_report_path = self.generate_data_report()
            report_paths["data_report"] = data_report_path
        
        # Generate training report if enabled
        if "training" in enabled_reports and self.training_results:
            training_report_path = self.generate_training_report()
            report_paths["training_report"] = training_report_path
        
        # Generate evaluation report if enabled
        if "evaluation" in enabled_reports and self.evaluation_results:
            evaluation_report_path = self.generate_evaluation_report()
            report_paths["evaluation_report"] = evaluation_report_path
        
        # Generate HTML summary report if enabled
        if "summary" in enabled_reports:
            summary_report_path = self.generate_summary_report(report_paths)
            report_paths["summary_report"] = summary_report_path
        
        logger.info(f"Generated {len(report_paths)} reports")
        return report_paths
    
    def generate_data_report(self) -> str:
        """
        Generate data distribution report.
        
        Returns:
            Path to the data report
        """
        logger.info("Generating data distribution report")
        
        if not self.data_manager:
            logger.warning("No data manager provided, cannot generate data report")
            return ""
        
        # Create figure for data distributions
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Data Distribution Report", fontsize=16)
        
        # Get train, val, test data
        try:
            X_train, y_train = self.data_manager.get_train_data()
            X_val, y_val = self.data_manager.get_val_data()
            X_test, y_test = self.data_manager.get_test_data()
            
            # Plot feature distributions
            self._plot_feature_distributions(fig, X_train, X_val, X_test)
            
            # Plot target distributions
            self._plot_target_distributions(fig, y_train, y_val, y_test)
            
        except Exception as e:
            logger.error(f"Error generating data distributions: {e}")
            
            # Plot simple data summary if available
            if self.data_manager.raw_data is not None:
                self._plot_data_summary(fig, self.data_manager.raw_data)
        
        # Save figure
        report_path = os.path.join(self.output_dir, "data_distribution_report.png")
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save data summary as JSON
        summary_path = os.path.join(self.output_dir, "data_summary.json")
        self._save_data_summary(summary_path)
        
        logger.info(f"Data distribution report saved to {report_path}")
        return report_path
    
    def _plot_feature_distributions(
        self, 
        fig: Figure, 
        X_train: np.ndarray, 
        X_val: np.ndarray, 
        X_test: np.ndarray
    ):
        """Plot feature distributions for train, val, and test sets."""
        n_features = X_train.shape[1]
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        feature_names = self._get_feature_names(n_features)
        
        for i in range(n_features):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            # Plot distributions
            sns.kdeplot(X_train[:, i], label="Train", ax=ax)
            sns.kdeplot(X_val[:, i], label="Validation", ax=ax)
            sns.kdeplot(X_test[:, i], label="Test", ax=ax)
            
            ax.set_title(feature_names[i])
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
    
    def _plot_target_distributions(
        self, 
        fig: Figure, 
        y_train: np.ndarray, 
        y_val: np.ndarray, 
        y_test: np.ndarray
    ):
        """Plot target distributions for train, val, and test sets."""
        n_targets = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        target_names = self._get_target_names(n_targets)
        
        # Determine subplot position
        n_features = len(self._get_feature_names(100))  # Get actual feature count
        n_cols = min(3, n_features)
        total_rows = (n_features + n_targets + n_cols - 1) // n_cols
        
        for i in range(n_targets):
            # Calculate subplot position
            subplot_idx = n_features + i + 1
            ax = fig.add_subplot(total_rows, n_cols, subplot_idx)
            
            # Extract target values
            if n_targets == 1 and len(y_train.shape) == 1:
                y_train_i = y_train
                y_val_i = y_val
                y_test_i = y_test
            else:
                y_train_i = y_train[:, i]
                y_val_i = y_val[:, i]
                y_test_i = y_test[:, i]
            
            # Plot distributions
            sns.kdeplot(y_train_i, label="Train", ax=ax)
            sns.kdeplot(y_val_i, label="Validation", ax=ax)
            sns.kdeplot(y_test_i, label="Test", ax=ax)
            
            ax.set_title(f"Target: {target_names[i]}")
            ax.set_xlabel("Value")
            ax.set_ylabel("Density")
            ax.legend()
    
    def _plot_data_summary(self, fig: Figure, data: Union[pd.DataFrame, np.ndarray]):
        """Plot summary of data when full distributions cannot be plotted."""
        if isinstance(data, pd.DataFrame):
            ax = fig.add_subplot(1, 1, 1)
            
            # Plot correlation heatmap
            corr = data.corr()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Feature Correlations")
            
        elif isinstance(data, np.ndarray):
            ax = fig.add_subplot(1, 1, 1)
            
            # Plot boxplot
            ax.boxplot(data)
            ax.set_title("Data Distribution")
            ax.set_xlabel("Feature Index")
            ax.set_ylabel("Value")
    
    def _save_data_summary(self, summary_path: str):
        """Save data summary as JSON."""
        if not self.data_manager:
            return
        
        # Get metadata from data manager
        data_metadata = self.data_manager.get_metadata()
        
        # Add additional summary statistics if available
        summary_stats = {}
        
        try:
            # Get train, val, test data
            X_train, y_train = self.data_manager.get_train_data()
            X_val, y_val = self.data_manager.get_val_data()
            X_test, y_test = self.data_manager.get_test_data()
            
            # Calculate summary statistics for features
            summary_stats["features"] = {
                "mean": np.mean(X_train, axis=0).tolist(),
                "std": np.std(X_train, axis=0).tolist(),
                "min": np.min(X_train, axis=0).tolist(),
                "max": np.max(X_train, axis=0).tolist(),
                "median": np.median(X_train, axis=0).tolist()
            }
            
            # Calculate summary statistics for targets
            if len(y_train.shape) == 1:
                y_train = y_train.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)
                y_test = y_test.reshape(-1, 1)
            
            summary_stats["targets"] = {
                "mean": np.mean(y_train, axis=0).tolist(),
                "std": np.std(y_train, axis=0).tolist(),
                "min": np.min(y_train, axis=0).tolist(),
                "max": np.max(y_train, axis=0).tolist(),
                "median": np.median(y_train, axis=0).tolist()
            }
        except Exception as e:
            logger.warning(f"Could not calculate summary statistics: {e}")
        
        # Combine metadata and summary statistics
        data_summary = {
            "metadata": data_metadata,
            "summary_stats": summary_stats,
            "timestamp": datetime.datetime.now().isoformat()
        }
        
        # Save to JSON
        with open(summary_path, 'w') as f:
            json.dump(data_summary, f, indent=2, default=str)
    
    def generate_training_report(self) -> str:
        """
        Generate training performance report.
        
        Returns:
            Path to the training report
        """
        logger.info("Generating training performance report")
        
        if not self.training_results:
            logger.warning("No training results provided, cannot generate training report")
            return ""
        
        # Create figure for training metrics
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Training Performance Report", fontsize=16)
        
        # Plot learning curves
        history = self.training_results.get("training_history", [])
        if history:
            self._plot_learning_curves(fig, history)
        else:
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No training history available", 
                   ha='center', va='center', fontsize=14)
        
        # Save figure
        report_path = os.path.join(self.output_dir, "training_performance_report.png")
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save training summary as JSON
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.training_results, f, indent=2, default=str)
        
        logger.info(f"Training performance report saved to {report_path}")
        return report_path
    
    def _plot_learning_curves(self, fig: Figure, history: List[Dict[str, float]]):
        """Plot learning curves from training history."""
        # Extract metrics from history
        metrics = {}
        for key in history[0].keys():
            metrics[key] = [epoch.get(key, 0) for epoch in history]
        
        # Determine the metrics to plot
        plot_metrics = []
        
        # Loss metrics
        loss_metrics = [m for m in metrics.keys() if "loss" in m.lower()]
        if loss_metrics:
            plot_metrics.append(loss_metrics)
        
        # Accuracy metrics
        acc_metrics = [m for m in metrics.keys() if "acc" in m.lower()]
        if acc_metrics:
            plot_metrics.append(acc_metrics)
        
        # Other metrics
        other_metrics = [m for m in metrics.keys() 
                         if m not in sum(plot_metrics, []) and "epoch" not in m.lower()]
        if other_metrics:
            # Group similar metrics
            metric_groups = {}
            for metric in other_metrics:
                base_name = metric.split('_')[0] if '_' in metric else metric
                if base_name not in metric_groups:
                    metric_groups[base_name] = []
                metric_groups[base_name].append(metric)
            
            for group in metric_groups.values():
                if group:
                    plot_metrics.append(group)
        
        # Plot each metric group
        n_plots = len(plot_metrics)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        for i, metric_group in enumerate(plot_metrics):
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            
            for metric in metric_group:
                ax.plot(metrics[metric], label=metric)
            
            ax.set_title(f"Learning Curve: {metric_group[0].split('_')[0].capitalize()}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True)
    
    def generate_evaluation_report(self) -> str:
        """
        Generate model evaluation report.
        
        Returns:
            Path to the evaluation report
        """
        logger.info("Generating model evaluation report")
        
        if not self.evaluation_results:
            logger.warning("No evaluation results provided, cannot generate evaluation report")
            return ""
        
        # Create figure for evaluation metrics
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("Model Evaluation Report", fontsize=16)
        
        # Prepare subplots
        n_plots = 3  # Predictions, Residuals, Metrics
        ax1 = fig.add_subplot(2, 2, 1)
        ax2 = fig.add_subplot(2, 2, 2)
        ax3 = fig.add_subplot(2, 1, 2)
        
        # Check if we have test predictions
        if "y_pred" in self.evaluation_results and "y_true" in self.evaluation_results:
            y_pred = np.array(self.evaluation_results["y_pred"])
            y_true = np.array(self.evaluation_results["y_true"])
            
            # Plot predictions vs true values
            self._plot_predictions(ax1, y_true, y_pred)
            
            # Plot residuals
            self._plot_residuals(ax2, y_true, y_pred)
        else:
            ax1.text(0.5, 0.5, "No prediction data available", 
                    ha='center', va='center', fontsize=12)
            ax2.text(0.5, 0.5, "No residual data available", 
                    ha='center', va='center', fontsize=12)
        
        # Plot metrics as a table
        self._plot_metrics_table(ax3, self.evaluation_results)
        
        # Save figure
        report_path = os.path.join(self.output_dir, "model_evaluation_report.png")
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # Save evaluation summary as JSON
        summary_path = os.path.join(self.output_dir, "evaluation_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(self.evaluation_results, f, indent=2, default=str)
        
        logger.info(f"Model evaluation report saved to {report_path}")
        return report_path
    
    def _plot_predictions(self, ax, y_true, y_pred):
        """Plot predictions vs true values."""
        ax.scatter(y_true, y_pred, alpha=0.5)
        
        # Add diagonal line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        ax.set_title("Predictions vs True Values")
        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.grid(True)
    
    def _plot_residuals(self, ax, y_true, y_pred):
        """Plot residuals."""
        residuals = y_true - y_pred
        
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        
        ax.set_title("Residuals vs Predicted Values")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.grid(True)
    
    def _plot_metrics_table(self, ax, evaluation_results):
        """Plot evaluation metrics as a table."""
        # Filter out non-metric entries
        metrics = {k: v for k, v in evaluation_results.items() 
                  if isinstance(v, (int, float)) and k not in ["y_true", "y_pred"]}
        
        if not metrics:
            ax.text(0.5, 0.5, "No metrics available", 
                   ha='center', va='center', fontsize=14)
            return
        
        # Create table data
        metric_names = list(metrics.keys())
        metric_values = [metrics[name] for name in metric_names]
        
        # Format table data
        table_data = [["Metric", "Value"]]
        for name, value in zip(metric_names, metric_values):
            formatted_name = name.replace("_", " ").capitalize()
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            table_data.append([formatted_name, formatted_value])
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colWidths=[0.6, 0.3],
            loc='center',
            cellLoc='center'
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        
        # Hide axes
        ax.axis('off')
        ax.set_title("Evaluation Metrics", fontsize=14)
    
    def generate_summary_report(self, report_paths: Dict[str, str]) -> str:
        """
        Generate HTML summary report.
        
        Args:
            report_paths: Dictionary of report paths
            
        Returns:
            Path to the summary report
        """
        logger.info("Generating HTML summary report")
        
        # Create HTML content
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>ML Pipeline Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                .section { margin-bottom: 30px; }
                .metrics-table { border-collapse: collapse; width: 100%; }
                .metrics-table th, .metrics-table td { border: 1px solid #ddd; padding: 8px; }
                .metrics-table tr:nth-child(even) { background-color: #f2f2f2; }
                .metrics-table th { padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #4CAF50; color: white; }
                .report-image { max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }
            </style>
        </head>
        <body>
            <h1>Machine Learning Pipeline Report</h1>
            <p>Generated on: {timestamp}</p>
            
            <div class="section">
                <h2>Overview</h2>
                {overview}
            </div>
            
            <div class="section">
                <h2>Data Summary</h2>
                {data_summary}
                <img class="report-image" src="{data_report_path}" alt="Data Distribution Report">
            </div>
            
            <div class="section">
                <h2>Training Performance</h2>
                {training_summary}
                <img class="report-image" src="{training_report_path}" alt="Training Performance Report">
            </div>
            
            <div class="section">
                <h2>Model Evaluation</h2>
                {evaluation_summary}
                <img class="report-image" src="{evaluation_report_path}" alt="Model Evaluation Report">
            </div>
        </body>
        </html>
        """
        
        # Generate report content
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Overview section
        overview = "<p>This report summarizes the results of the machine learning pipeline.</p>"
        if self.config.get("experiment_name"):
            overview += f"<p>Experiment: <strong>{self.config.get('experiment_name')}</strong></p>"
        
        # Data summary section
        data_summary = "<p>No data summary available.</p>"
        if self.data_manager:
            metadata = self.data_manager.get_metadata()
            data_summary = f"""
            <table class="metrics-table">
                <tr><th>Item</th><th>Value</th></tr>
                <tr><td>Total Samples</td><td>{metadata.get('total_samples', 'N/A')}</td></tr>
            """
            
            # Add splitting info if available
            splitting = metadata.get("splitting", {})
            if splitting:
                train_shape = splitting.get("train_shape", {})
                val_shape = splitting.get("val_shape", {})
                test_shape = splitting.get("test_shape", {})
                
                data_summary += f"""
                <tr><td>Train Set Size</td><td>{train_shape.get('rows', 'N/A')}</td></tr>
                <tr><td>Validation Set Size</td><td>{val_shape.get('rows', 'N/A')}</td></tr>
                <tr><td>Test Set Size</td><td>{test_shape.get('rows', 'N/A')}</td></tr>
                """
            
            data_summary += "</table>"
        
        # Training summary section
        training_summary = "<p>No training summary available.</p>"
        if self.training_results:
            tr = self.training_results
            training_summary = f"""
            <table class="metrics-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Epochs Completed</td><td>{tr.get('epochs_completed', 'N/A')}</td></tr>
                <tr><td>Best Validation Loss</td><td>{tr.get('best_val_loss', 'N/A'):.4f}</td></tr>
                <tr><td>Training Time</td><td>{tr.get('training_time', 'N/A'):.2f} seconds</td></tr>
            </table>
            """
        
        # Evaluation summary section
        evaluation_summary = "<p>No evaluation summary available.</p>"
        if self.evaluation_results:
            er = self.evaluation_results
            evaluation_summary = "<table class='metrics-table'><tr><th>Metric</th><th>Value</th></tr>"
            
            # Add each metric to the table
            for key, value in er.items():
                if isinstance(value, (int, float)) and key not in ["y_true", "y_pred"]:
                    formatted_key = key.replace("_", " ").capitalize()
                    formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
                    evaluation_summary += f"<tr><td>{formatted_key}</td><td>{formatted_value}</td></tr>"
            
            evaluation_summary += "</table>"
        
        # Format HTML with all sections
        html_content = html_content.format(
            timestamp=timestamp,
            overview=overview,
            data_summary=data_summary,
            training_summary=training_summary,
            evaluation_summary=evaluation_summary,
            data_report_path=os.path.basename(report_paths.get("data_report", "")),
            training_report_path=os.path.basename(report_paths.get("training_report", "")),
            evaluation_report_path=os.path.basename(report_paths.get("evaluation_report", ""))
        )
        
        # Save HTML report
        report_path = os.path.join(self.output_dir, "summary_report.html")
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML summary report saved to {report_path}")
        return report_path
    
    def _get_feature_names(self, n_features: int) -> List[str]:
        """Get feature names from configuration or generate generic names."""
        feature_names = self.config.get("data", {}).get("feature_names", [])
        
        if feature_names and len(feature_names) >= n_features:
            return feature_names[:n_features]
        
        # Generate generic feature names
        return [f"Feature {i+1}" for i in range(n_features)]
    
    def _get_target_names(self, n_targets: int) -> List[str]:
        """Get target names from configuration or generate generic names."""
        target_names = self.config.get("data", {}).get("target_names", [])
        
        if target_names and len(target_names) >= n_targets:
            return target_names[:n_targets]
        
        # Generate generic target names
        return [f"Target {i+1}" for i in range(n_targets)]
