#!/usr/bin/env python3
"""MNIST report generator plugin for the ML Provenance Tracker Framework."""

import matplotlib.pyplot as plt
import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional

from src.tof_ml.reporting.report_generator import ClassificationReportGenerator
from src.tof_ml.pipeline.plugins.interfaces import ReportGeneratorPlugin
from src.tof_ml.data.data_manager import DataManager

logger = logging.getLogger(__name__)


class MNISTReportGenerator(ClassificationReportGenerator, ReportGeneratorPlugin):
    """
    Report generator plugin for MNIST dataset.
    Extends the classification report generator with MNIST-specific visualizations.
    """

    def __init__(
            self,
            config: Dict[str, Any],
            data_manager: Optional[DataManager] = None,
            training_results: Optional[Dict[str, Any]] = None,
            evaluation_results: Optional[Dict[str, Any]] = None,
            output_dir: str = "./reports"
    ):
        """Initialize the MNIST report generator."""
        super().__init__(config, data_manager, training_results, evaluation_results, output_dir)
        logger.info("MNISTReportGenerator initialized")

    def generate_data_report(self) -> str:
        """Generate data distribution report specialized for MNIST."""
        logger.info("Generating MNIST data report")

        if not self.data_manager:
            logger.warning("No data manager provided, cannot generate data report")
            return ""

        # Create figure for data distributions
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("MNIST Data Distribution Report", fontsize=16)

        # Get train data
        try:
            X_train, y_train = self.data_manager.get_train_data()

            # Plot sample images
            self._plot_mnist_samples(fig, X_train, y_train)

            # Plot digit class distribution
            ax = fig.add_subplot(2, 1, 2)
            self._plot_class_distribution(ax, y_train,
                                          self.data_manager.get_val_data()[1],
                                          self.data_manager.get_test_data()[1])

        except Exception as e:
            logger.error(f"Error generating MNIST visualizations: {e}")

            # Fallback visualization
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error generating visualizations: {e}",
                    ha='center', va='center', fontsize=12)

        # Save figure
        report_path = os.path.join(self.output_dir, "mnist_data_report.png")
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Save data summary as JSON
        summary_path = os.path.join(self.output_dir, "mnist_data_summary.json")
        self._save_metadata(summary_path, self._get_mnist_data_summary())

        logger.info(f"MNIST data report saved to {report_path}")
        return report_path

    def generate_training_report(self) -> str:
        """Generate training performance report for MNIST."""
        logger.info("Generating MNIST training report")

        if not self.training_results:
            logger.warning("No training results provided, cannot generate training report")
            return ""

        # Create figure for training performance
        fig = plt.figure(figsize=(15, 8))
        fig.suptitle("MNIST Training Performance", fontsize=16)

        # Plot training curves
        history = self.training_results.get("training_history", {})
        if history:
            # Plot loss curves
            ax1 = fig.add_subplot(1, 2, 1)
            if "loss" in history:
                ax1.plot(history.get("loss", []), label="Training Loss")
            if "val_loss" in history:
                ax1.plot(history.get("val_loss", []), label="Validation Loss")
            ax1.set_title("Loss Curves")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True)

            # Plot accuracy curves
            ax2 = fig.add_subplot(1, 2, 2)
            if "accuracy" in history:
                ax2.plot(history.get("accuracy", []), label="Training Accuracy")
            if "val_accuracy" in history:
                ax2.plot(history.get("val_accuracy", []), label="Validation Accuracy")
            ax2.set_title("Accuracy Curves")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True)
        else:
            # Show message if no history available
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No training history available",
                    ha='center', va='center', fontsize=14)

        # Save figure
        report_path = os.path.join(self.output_dir, "mnist_training_report.png")
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        logger.info(f"MNIST training report saved to {report_path}")
        return report_path

    def generate_evaluation_report(self) -> str:
        """Generate model evaluation report for MNIST."""
        logger.info("Generating MNIST evaluation report")

        if not self.evaluation_results:
            logger.warning("No evaluation results provided, cannot generate evaluation report")
            return ""

        # Create figure for evaluation visualizations
        fig = plt.figure(figsize=(15, 12))
        fig.suptitle("MNIST Evaluation Results", fontsize=16)

        try:
            # Get predictions and true values from evaluation results
            y_true = np.array(self.evaluation_results.get("y_true", []))
            y_pred = np.array(self.evaluation_results.get("y_pred", []))
            
            # Get confusion matrix if available
            confusion_matrix = np.array(self.evaluation_results.get("confusion_matrix", []))
            
            # Plot confusion matrix
            if confusion_matrix.size > 0:
                ax1 = fig.add_subplot(2, 1, 1)
                self._plot_confusion_matrix(ax1, confusion_matrix)
            
            # Plot misclassified examples
            ax2 = fig.add_subplot(2, 1, 2)
            X_test, _ = self.data_manager.get_test_data()
            self._plot_misclassified_digits(ax2, X_test, y_true, y_pred)
            
        except Exception as e:
            logger.error(f"Error generating MNIST evaluation visualizations: {e}")
            
            # Fallback visualization
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, f"Error generating visualizations: {e}",
                    ha='center', va='center', fontsize=12)

        # Save figure
        report_path = os.path.join(self.output_dir, "mnist_evaluation_report.png")
        fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the suptitle
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Save evaluation metrics summary
        metrics_path = os.path.join(self.output_dir, "mnist_evaluation_metrics.json")
        metrics_summary = {k: v for k, v in self.evaluation_results.items() 
                           if k not in ["y_true", "y_pred", "confusion_matrix"]}
        self._save_metadata(metrics_path, metrics_summary)

        logger.info(f"MNIST evaluation report saved to {report_path}")
        return report_path

    def _plot_mnist_samples(self, fig, X_data, y_data):
        """Plot sample MNIST digits with their labels."""
        n_samples = min(10, X_data.shape[0])

        # Create subplot grid
        for i in range(n_samples):
            ax = fig.add_subplot(2, 5, i + 1)

            # Handle both flattened (784,) and CNN format (28, 28, 1)
            if len(X_data.shape) == 2:  # Flattened format
                img = X_data[i].reshape(28, 28)
            else:  # CNN format (28, 28, 1)
                img = X_data[i].squeeze()  # Remove the channel dimension

            ax.imshow(img, cmap='gray')
            # Handle both 1D and 2D label formats
            label = int(y_data[i]) if y_data[i].size == 1 else int(y_data[i, 0])
            ax.set_title(f"Label: {label}")
            ax.axis('off')

    def _plot_confusion_matrix(self, ax, confusion_matrix):
        """Plot confusion matrix for MNIST classification."""
        im = ax.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Show all ticks
        n_classes = confusion_matrix.shape[0]
        ax.set_xticks(np.arange(n_classes))
        ax.set_yticks(np.arange(n_classes))
        
        # Label with class indices
        ax.set_xticklabels(np.arange(n_classes))
        ax.set_yticklabels(np.arange(n_classes))
        
        # Set axis labels
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title('Confusion Matrix')
        
        # Loop over data dimensions and create text annotations
        max_val = confusion_matrix.max()
        thresh = max_val / 2. if max_val > 0 else 0
        for i in range(n_classes):
            for j in range(n_classes):
                cell_val = int(confusion_matrix[i, j])
                ax.text(j, i, str(cell_val),
                        ha="center", va="center",
                        color="white" if cell_val > thresh else "black")

    def _plot_misclassified_digits(self, ax, X_test, y_true, y_pred):
        """Plot examples of misclassified digits."""
        # Find misclassified examples
        misclassified = np.where(y_true != y_pred)[0]

        if len(misclassified) == 0:
            ax.text(0.5, 0.5, "No misclassified examples found",
                    ha='center', va='center', fontsize=14)
            return

        # Sample up to 10 misclassified examples
        n_samples = min(10, len(misclassified))
        indices = np.random.choice(misclassified, n_samples, replace=False)

        # Create a grid of subplots
        for i, idx in enumerate(indices):
            # Create subplot within the main subplot area
            ax_sub = ax.inset_axes([i / n_samples, 0, 1 / n_samples, 1])

            # Handle both flattened (784,) and CNN format (28, 28, 1)
            if len(X_test.shape) == 2:  # Flattened format
                img = X_test[idx].reshape(28, 28)
            else:  # CNN format (28, 28, 1)
                img = X_test[idx].squeeze()  # Remove the channel dimension

            ax_sub.imshow(img, cmap='gray')
            ax_sub.set_title(f"True: {int(y_true[idx])}\nPred: {int(y_pred[idx])}")
            ax_sub.axis('off')

        ax.axis('off')
        ax.set_title("Misclassified Examples", pad=20)

    def _get_mnist_data_summary(self) -> Dict[str, Any]:
        """Get MNIST-specific data summary."""
        summary = {}

        if self.data_manager:
            X_train, y_train = self.data_manager.get_train_data()
            X_val, y_val = self.data_manager.get_val_data()
            X_test, y_test = self.data_manager.get_test_data()

            # Dataset sizes
            summary["train_size"] = len(X_train)
            summary["val_size"] = len(X_val)
            summary["test_size"] = len(X_test)

            # Class distribution
            unique_classes = np.unique(np.concatenate([y_train, y_val, y_test]))
            class_counts = {}
            for c in unique_classes:
                class_counts[f"class_{c}"] = {
                    "train": int(np.sum(y_train == c)),
                    "val": int(np.sum(y_val == c)),
                    "test": int(np.sum(y_test == c))
                }
            summary["class_distribution"] = class_counts

            # Data shape
            if len(X_train.shape) == 2:  # Flattened format
                summary["image_shape"] = "28x28 (flattened)"
                summary["flattened"] = True
            else:  # CNN format
                summary["image_shape"] = f"{X_train.shape[1]}x{X_train.shape[2]}x{X_train.shape[3]}"
                summary["flattened"] = False

        return summary