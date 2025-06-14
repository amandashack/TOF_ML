#!/usr/bin/env python3
"""MNIST report generator plugin for the ML Provenance Tracker Framework."""

import matplotlib.pyplot as plt
import os
import logging
import numpy as np
from typing import Dict, Any, List, Optional
import seaborn as sns
import pandas as pd

from src.tof_ml.reporting.report_generator import ReportGeneratorPlugin, ClassificationReportMixin
from src.tof_ml.data.data_manager import DataManager
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score

logger = logging.getLogger(__name__)


class MNISTReportGenerator(ClassificationReportMixin, ReportGeneratorPlugin):
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
        fig, axes = plt.subplots(2, 1, figsize=(12, 16))
        fig.suptitle("MNIST Evaluation Results", fontsize=16)

        try:
            # Get predictions and true values from evaluation results
            y_true = np.array(self.evaluation_results.get("y_true", []))
            y_pred = np.array(self.evaluation_results.get("y_pred", []))
            
            # Get confusion matrix if available
            confusion_matrix = np.array(self.evaluation_results.get("confusion_matrix", []))
            print(confusion_matrix, type(confusion_matrix), type(confusion_matrix[0]), type(confusion_matrix[0][0]))

            # Plot confusion matrix
            if confusion_matrix.size > 0:
                self._plot_confusion_matrix(axes[0], confusion_matrix)
            
            """# Plot misclassified examples
            ax2 = fig.add_subplot(2, 1, 2)
            X_test, _ = self.data_manager.get_test_data()
            self._plot_misclassified_digits(axes[1], X_test, y_true, y_pred)"""
            
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
        """Plot confusion matrix for MNIST classification using seaborn."""
        # Ensure the confusion matrix is a proper numpy array
        confusion_matrix = np.asarray(confusion_matrix)

        # Create DataFrame for seaborn heatmap
        # For MNIST, we have digits 0-9 as class labels
        class_labels = [str(i) for i in range(confusion_matrix.shape[0])]

        df_cm = pd.DataFrame(
            confusion_matrix,
            index=class_labels,
            columns=class_labels
        )

        # Create the heatmap
        sns.heatmap(
            df_cm,
            annot=True,  # Show numbers in cells
            fmt='d',  # Format as integers
            cmap='Blues',  # Use blue color scheme
            ax=ax,  # Use the provided axis
            square=True,  # Make cells square
            linewidths=0.5,  # Add lines between cells
            annot_kws={'size': 8},  # Smaller font size for better fit
            cbar_kws={'shrink': 0.8}  # Shrink colorbar to fit better
        )

        # Set labels and title
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, pad=10)

        # Ensure labels are readable
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)


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


class MNISTConvReportGenerator(ReportGeneratorPlugin, ClassificationReportMixin):
    """
    Report generator for MNIST classification.
    Inherits base functionality from ReportGeneratorPlugin and
    classification utilities from ClassificationReportMixin.
    """

    def generate_data_report(self) -> str:
        """Generate MNIST data distribution report."""
        logger.info("Generating MNIST data report")

        fig = plt.figure(figsize=(20, 10))
        fig.suptitle("MNIST Dataset Analysis", fontsize=16)

        # Get data
        X_train, y_train = self.data_manager.get_train_data()
        X_val, y_val = self.data_manager.get_val_data()
        X_test, y_test = self.data_manager.get_test_data()

        # Use mixin method for class distribution
        ax1 = fig.add_subplot(2, 3, 1)
        self._plot_class_distribution(ax1, y_train, y_val, y_test)

        # Add MNIST-specific visualizations
        ax2 = fig.add_subplot(2, 3, 2)
        self._plot_sample_digits(ax2, X_train, y_train)

        # Dataset statistics
        ax5 = fig.add_subplot(2, 3, 5)
        self._plot_dataset_stats(ax5, X_train, X_val, X_test)

        report_path = os.path.join(self.output_dir, "mnist_data_report.png")
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return report_path

    def generate_training_report(self) -> str:
        """Generate CNN training performance report."""
        logger.info("Generating MNIST training report")

        if not self.training_results:
            logger.warning("No training results provided")
            return ""

        # Create figure
        fig = plt.figure(figsize=(20, 12))
        fig.suptitle("MNIST CNN Training Report", fontsize=16)

        history = self.training_results.get("training_history", {})

        if history:
            # 1. Loss curves
            ax1 = fig.add_subplot(2, 3, 1)
            self._plot_loss_curves(ax1, history)

            # 2. Accuracy curves
            ax2 = fig.add_subplot(2, 3, 2)
            self._plot_accuracy_curves(ax2, history)

            # 3. Learning rate schedule (if available)
            ax3 = fig.add_subplot(2, 3, 3)
            self._plot_learning_rate(ax3, history)

            # 4. Training summary
            ax4 = fig.add_subplot(2, 3, 4)
            self._plot_training_summary(ax4)

            # 6. Training time analysis
            ax6 = fig.add_subplot(2, 3, 6)
            self._plot_training_time(ax6)

        # Save figure
        report_path = os.path.join(self.output_dir, "mnist_training_report.png")
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return report_path

    def generate_evaluation_report(self) -> str:
        """Generate CNN evaluation report."""
        logger.info("Generating MNIST evaluation report")

        if not self.evaluation_results:
            logger.warning("No evaluation results provided")
            return ""

        # Create figure
        fig = plt.figure(figsize=(20, 15))
        fig.suptitle("MNIST CNN Evaluation Report", fontsize=16)

        # Get predictions and true labels
        y_true = np.array(self.evaluation_results.get("y_true", []))
        y_pred_probs = np.array(self.evaluation_results.get("y_pred", []))

        if len(y_pred_probs.shape) > 1:
            y_pred = np.argmax(y_pred_probs, axis=1)
        else:
            y_pred = y_pred_probs.astype(int)

        # 1. Confusion matrix
        ax1 = fig.add_subplot(3, 3, 1)
        self._plot_confusion_matrix(ax1, y_true, y_pred)

        # 2. Per-class metrics
        ax2 = fig.add_subplot(3, 3, 2)
        self._plot_per_class_metrics(ax2, y_true, y_pred)

        # 3. Misclassified samples
        ax3 = fig.add_subplot(3, 3, 3)
        self._plot_misclassified_samples(ax3, y_true, y_pred)

        # 4. Confidence distribution
        ax4 = fig.add_subplot(3, 3, 4)
        if len(y_pred_probs.shape) > 1:
            self._plot_confidence_distribution(ax4, y_pred_probs)

        # 5. ROC curves (one-vs-rest)
        ax5 = fig.add_subplot(3, 3, 5)
        if len(y_pred_probs.shape) > 1:
            self._plot_roc_curves(ax5, y_true, y_pred_probs)

        # 7. Error analysis
        ax7 = fig.add_subplot(3, 3, 7)
        self._plot_error_analysis(ax7, y_true, y_pred)

        # 6. Prediction distribution
        ax6 = fig.add_subplot(3, 3, 6)
        self._plot_prediction_distribution(ax6, y_pred)

        # Save figure
        report_path = os.path.join(self.output_dir, "mnist_evaluation_report.png")
        fig.savefig(report_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        # Generate text report
        self._generate_text_report(y_true, y_pred)

        return report_path

    # Add these methods to your MNISTConvReportGenerator class:

    def _plot_sample_digits(self, ax, X_data, y_data):
        """Plot sample MNIST digits with their labels."""
        n_samples = min(10, X_data.shape[0])

        # Clear the axis and set up for subplots
        ax.axis('off')
        ax.set_title("Sample Digits")

        # Create a grid of sample images
        for i in range(n_samples):
            # Create inset axes for each image
            left = (i % 5) * 0.2
            bottom = 0.5 if i < 5 else 0.0
            width = 0.18
            height = 0.4

            ax_img = ax.inset_axes([left, bottom, width, height])

            # Handle both flattened (784,) and CNN format (28, 28, 1)
            if len(X_data.shape) == 2:  # Flattened format
                img = X_data[i].reshape(28, 28)
            else:  # CNN format (28, 28, 1)
                img = X_data[i].squeeze()  # Remove the channel dimension

            ax_img.imshow(img, cmap='gray')
            # Handle both 1D and 2D label formats
            label = int(y_data[i]) if y_data[i].size == 1 else int(y_data[i, 0])
            ax_img.set_title(f"{label}", fontsize=8)
            ax_img.axis('off')

    def _plot_dataset_stats(self, ax, X_train, X_val, X_test):
        """Plot dataset statistics."""
        stats = {
            'Training': len(X_train),
            'Validation': len(X_val),
            'Test': len(X_test)
        }

        ax.bar(stats.keys(), stats.values())
        ax.set_title('Dataset Size Distribution')
        ax.set_ylabel('Number of Samples')

        # Add value labels on bars
        for i, (name, value) in enumerate(stats.items()):
            ax.text(i, value + 100, str(value), ha='center', va='bottom')

    def _plot_loss_curves(self, ax, history):
        """Plot training and validation loss curves."""
        if "loss" in history:
            ax.plot(history["loss"], label="Training Loss")
        if "val_loss" in history:
            ax.plot(history["val_loss"], label="Validation Loss")

        ax.set_title("Loss Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_accuracy_curves(self, ax, history):
        """Plot training and validation accuracy curves."""
        if "accuracy" in history:
            ax.plot(history["accuracy"], label="Training Accuracy")
        if "val_accuracy" in history:
            ax.plot(history["val_accuracy"], label="Validation Accuracy")

        ax.set_title("Accuracy Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_learning_rate(self, ax, history):
        """Plot learning rate schedule if available."""
        if "lr" in history:
            ax.plot(history["lr"])
            ax.set_title("Learning Rate Schedule")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Learning Rate")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No learning rate data available",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Learning Rate Schedule")

    def _plot_training_summary(self, ax):
        """Plot training summary statistics."""
        if not self.training_results:
            ax.text(0.5, 0.5, "No training data available",
                    ha='center', va='center', transform=ax.transAxes)
            return

        # Create summary statistics
        summary_data = {
            'Epochs': self.training_results.get('epochs_completed', 0),
            'Best Val Loss': self.training_results.get('best_val_loss', 0),
            'Training Time (min)': self.training_results.get('training_time', 0) / 60
        }

        # Create a simple text summary
        ax.axis('off')
        ax.set_title("Training Summary")

        y_pos = 0.8
        for key, value in summary_data.items():
            if isinstance(value, float):
                text = f"{key}: {value:.4f}"
            else:
                text = f"{key}: {value}"
            ax.text(0.1, y_pos, text, transform=ax.transAxes, fontsize=12)
            y_pos -= 0.2

    def _plot_training_time(self, ax):
        """Plot training time analysis."""
        if not self.training_results:
            ax.text(0.5, 0.5, "No training data available",
                    ha='center', va='center', transform=ax.transAxes)
            return

        training_time = self.training_results.get('training_time', 0)
        epochs = self.training_results.get('epochs_completed', 1)

        ax.bar(['Total Time', 'Time per Epoch'],
               [training_time, training_time / epochs])
        ax.set_title("Training Time Analysis")
        ax.set_ylabel("Time (seconds)")

    def _plot_misclassified_samples(self, ax, y_true, y_pred):
        """Plot misclassified samples."""
        # Find misclassified examples
        misclassified = np.where(y_true != y_pred)[0]

        if len(misclassified) == 0:
            ax.text(0.5, 0.5, "No misclassified examples found",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title("Misclassified Examples")
            return

        ax.text(0.5, 0.5,
                f"Found {len(misclassified)} misclassified examples\n({len(misclassified) / len(y_true) * 100:.1f}%)",
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title("Misclassified Examples")
        ax.axis('off')

    def _plot_confidence_distribution(self, ax, y_pred_probs):
        """Plot confidence distribution."""
        if len(y_pred_probs.shape) > 1:
            # Get max probability for each prediction
            confidences = np.max(y_pred_probs, axis=1)
            ax.hist(confidences, bins=50, alpha=0.7, edgecolor='black')
            ax.set_title("Prediction Confidence Distribution")
            ax.set_xlabel("Confidence (Max Probability)")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No probability data available",
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_roc_curves(self, ax, y_true, y_pred_probs):
        """Plot ROC curves for multiclass classification."""
        try:
            from sklearn.metrics import roc_curve, auc
            from sklearn.preprocessing import label_binarize

            # Binarize the labels
            y_bin = label_binarize(y_true, classes=range(10))

            # Plot ROC curve for each class
            for i in range(min(3, 10)):  # Show only first 3 classes to avoid clutter
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_probs[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})')

            ax.plot([0, 1], [0, 1], 'k--')
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves (First 3 Classes)')
            ax.legend()

        except ImportError:
            ax.text(0.5, 0.5, "sklearn required for ROC curves",
                    ha='center', va='center', transform=ax.transAxes)

    def _plot_prediction_distribution(self, ax, y_pred):
        """Plot distribution of predictions."""
        unique_classes, counts = np.unique(y_pred, return_counts=True)

        ax.bar(unique_classes, counts)
        ax.set_title("Prediction Distribution")
        ax.set_xlabel("Predicted Class")
        ax.set_ylabel("Count")
        ax.set_xticks(unique_classes)

    def _plot_error_analysis(self, ax, y_true, y_pred):
        """Plot error analysis."""
        # Calculate per-class error rates
        error_rates = []
        classes = np.unique(y_true)

        for cls in classes:
            mask = y_true == cls
            if np.sum(mask) > 0:
                error_rate = np.sum(y_true[mask] != y_pred[mask]) / np.sum(mask)
                error_rates.append(error_rate)
            else:
                error_rates.append(0)

        ax.bar(classes, error_rates)
        ax.set_title("Per-Class Error Rates")
        ax.set_xlabel("Class")
        ax.set_ylabel("Error Rate")
        ax.set_xticks(classes)
        ax.grid(True, alpha=0.3)

    def _generate_text_report(self, y_true, y_pred):
        """Generate text classification report."""
        try:
            from sklearn.metrics import classification_report
            report = classification_report(y_true, y_pred)

            # Save text report
            report_path = os.path.join(self.output_dir, "classification_report.txt")
            with open(report_path, 'w') as f:
                f.write("MNIST Classification Report\n")
                f.write("=" * 50 + "\n\n")
                f.write(report)

            logger.info(f"Text classification report saved to {report_path}")

        except ImportError:
            logger.warning("sklearn not available for detailed classification report")