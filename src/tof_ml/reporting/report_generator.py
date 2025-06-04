#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plugin interfaces for report generators with mixins for specific report types.
"""

from abc import ABC, abstractmethod
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import logging
import json
from typing import Dict, Any, Optional, List
from sklearn.metrics import confusion_matrix, classification_report

logger = logging.getLogger(__name__)

class ReportGeneratorPlugin(ABC):
    """
    Base abstract class for all report generator plugins.
    Provides core functionality and enforces the plugin contract.
    """

    def __init__(
            self,
            config: Dict[str, Any],
            data_manager: Optional[Any] = None,
            training_results: Optional[Dict[str, Any]] = None,
            evaluation_results: Optional[Dict[str, Any]] = None,
            output_dir: str = "./reports"
    ):
        """Initialize the report generator."""
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

    def _setup_plotting_style(self):
        """Set up matplotlib plotting style."""
        style = self.report_config.get("style", "tableau-colorblind10")
        plt.style.use(style)
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10

    @abstractmethod
    def generate_data_report(self) -> str:
        """Generate data distribution report."""
        pass

    @abstractmethod
    def generate_training_report(self) -> str:
        """Generate training performance report."""
        pass

    @abstractmethod
    def generate_evaluation_report(self) -> str:
        """Generate model evaluation report."""
        pass

    def generate_reports(self) -> Dict[str, str]:
        """
        Generate all reports based on configuration.
        This is a template method that calls the specific report generators.
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

    def generate_summary_report(self, report_paths: Dict[str, str]) -> str:
        """
        Generate HTML summary report.
        This is a concrete implementation that can be overridden if needed.
        """
        logger.info("Generating HTML summary report")

        # Create HTML content template
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>ML Pipeline Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                .section {{ margin-bottom: 30px; }}
                .metrics-table {{ border-collapse: collapse; width: 100%; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; }}
                .metrics-table tr:nth-child(even) {{ background-color: #f2f2f2; }}
                .metrics-table th {{ padding-top: 12px; padding-bottom: 12px; text-align: left; background-color: #4CAF50; color: white; }}
                .report-image {{ max-width: 100%; height: auto; border: 1px solid #ddd; margin-top: 10px; }}
            </style>
        </head>
        <body>
            <h1>Machine Learning Pipeline Report</h1>
            <p>Generated on: {timestamp}</p>

            <div class="section">
                <h2>Overview</h2>
                {overview}
            </div>

            {data_section}

            {training_section}

            {evaluation_section}
        </body>
        </html>
        """

        # Generate report content
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Overview section
        overview = self._generate_overview_section()

        # Generate sections based on available reports
        data_section = self._generate_data_section(report_paths.get("data_report", ""))
        training_section = self._generate_training_section(report_paths.get("training_report", ""))
        evaluation_section = self._generate_evaluation_section(report_paths.get("evaluation_report", ""))

        # Format HTML with all sections
        html_content = html_content.format(
            timestamp=timestamp,
            overview=overview,
            data_section=data_section,
            training_section=training_section,
            evaluation_section=evaluation_section
        )

        # Save HTML report
        report_path = os.path.join(self.output_dir, "summary_report.html")
        with open(report_path, 'w') as f:
            f.write(html_content)

        logger.info(f"HTML summary report saved to {report_path}")
        return report_path

    def _generate_overview_section(self) -> str:
        """Generate the overview section for summary report."""
        overview = "<p>This report summarizes the results of the machine learning pipeline.</p>"
        if self.config.get("experiment_name"):
            overview += f"<p>Experiment: <strong>{self.config.get('experiment_name')}</strong></p>"
        return overview

    def _generate_data_section(self, data_report_path: str) -> str:
        """Generate the data section for summary report."""
        if not data_report_path:
            return ""

        data_summary = "<p>No data summary available.</p>"
        if self.data_manager:
            metadata = self.data_manager.get_data_summary()
            data_summary = self._format_data_summary_table(metadata)

        return f"""
        <div class="section">
            <h2>Data Summary</h2>
            {data_summary}
            <img class="report-image" src="{os.path.basename(data_report_path)}" alt="Data Distribution Report">
        </div>
        """

    def _generate_training_section(self, training_report_path: str) -> str:
        """Generate the training section for summary report."""
        if not training_report_path:
            return ""

        training_summary = "<p>No training summary available.</p>"
        if self.training_results:
            training_summary = self._format_training_summary_table()

        return f"""
        <div class="section">
            <h2>Training Performance</h2>
            {training_summary}
            <img class="report-image" src="{os.path.basename(training_report_path)}" alt="Training Performance Report">
        </div>
        """

    def _generate_evaluation_section(self, evaluation_report_path: str) -> str:
        """Generate the evaluation section for summary report."""
        if not evaluation_report_path:
            return ""

        evaluation_summary = "<p>No evaluation summary available.</p>"
        if self.evaluation_results:
            evaluation_summary = self._format_evaluation_summary_table()

        return f"""
        <div class="section">
            <h2>Model Evaluation</h2>
            {evaluation_summary}
            <img class="report-image" src="{os.path.basename(evaluation_report_path)}" alt="Model Evaluation Report">
        </div>
        """

    def _format_data_summary_table(self, metadata: Dict[str, Any]) -> str:
        """Format data summary as HTML table."""
        # Implementation specific to each subclass
        return "<p>Data summary available in specific report.</p>"

    def _format_training_summary_table(self) -> str:
        """Format training summary as HTML table."""
        # Implementation specific to each subclass
        return "<p>Training summary available in specific report.</p>"

    def _format_evaluation_summary_table(self) -> str:
        """Format evaluation summary as HTML table."""
        # Implementation specific to each subclass
        return "<p>Evaluation summary available in specific report.</p>"

    def _save_metadata(self, path: str, metadata: Dict[str, Any]) -> None:
        """Save metadata to JSON file."""
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


class ClassificationReportMixin:
    """
    Mixin providing classification-specific plotting utilities.
    """

    def _plot_class_distribution(self, ax, y_train, y_val, y_test):
        """Plot class distribution for classification tasks."""
        unique_classes = np.unique(np.concatenate([y_train.flatten(), y_val.flatten(), y_test.flatten()]))

        train_counts = [np.sum(y_train.flatten() == c) for c in unique_classes]
        val_counts = [np.sum(y_val.flatten() == c) for c in unique_classes]
        test_counts = [np.sum(y_test.flatten() == c) for c in unique_classes]

        x = np.arange(len(unique_classes))
        width = 0.25

        ax.bar(x - width, train_counts, width, label='Train')
        ax.bar(x, val_counts, width, label='Validation')
        ax.bar(x + width, test_counts, width, label='Test')

        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.set_xticks(x)
        ax.set_xticklabels(unique_classes)
        ax.legend()

    def _plot_confusion_matrix(self, ax, y_true, y_pred, normalize=True):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                    cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')

    def _plot_per_class_metrics(self, ax, y_true, y_pred):
        """Plot per-class precision, recall, and F1 scores."""
        from sklearn.metrics import precision_recall_fscore_support

        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None
        )

        classes = np.arange(len(precision))
        x = np.arange(len(classes))
        width = 0.25

        ax.bar(x - width, precision, width, label='Precision')
        ax.bar(x, recall, width, label='Recall')
        ax.bar(x + width, f1, width, label='F1-Score')

        ax.set_xlabel('Class')
        ax.set_ylabel('Score')
        ax.set_title('Per-Class Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim([0, 1.05])

    def _generate_classification_text_report(self, y_true, y_pred, class_names=None):
        """Generate detailed classification report."""
        report = classification_report(y_true, y_pred, target_names=class_names)
        return report


class RegressionReportMixin:
    """
    Mixin providing regression-specific plotting utilities.
    """

    def _plot_predictions_vs_actual(self, ax, y_true, y_pred):
        """Plot predictions vs true values for regression."""
        ax.scatter(y_true, y_pred, alpha=0.5)

        # Add diagonal line
        min_val = min(np.min(y_true), np.min(y_pred))
        max_val = max(np.max(y_true), np.max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')

        ax.set_title("Predictions vs Actual Values")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_residuals(self, ax, y_true, y_pred):
        """Plot residuals for regression tasks."""
        residuals = y_true - y_pred

        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')

        ax.set_title("Residual Plot")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.grid(True, alpha=0.3)

    def _plot_residual_distribution(self, ax, y_true, y_pred):
        """Plot distribution of residuals."""
        residuals = y_true - y_pred

        ax.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', label='Zero Residual')
        ax.set_title("Residual Distribution")
        ax.set_xlabel("Residual Value")
        ax.set_ylabel("Frequency")
        ax.legend()

    def _calculate_regression_metrics(self, y_true, y_pred):
        """Calculate common regression metrics."""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }


class TimeSeriesReportMixin:
    """Mixin for time series specific visualizations."""

    def _plot_time_series_predictions(self, ax, timestamps, y_true, y_pred):
        """Plot time series predictions."""
        # ... implementation ...
        pass

    def _plot_seasonal_decomposition(self, ax, data):
        """Plot seasonal decomposition."""
        # ... implementation ...
        pass