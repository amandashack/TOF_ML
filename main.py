"""
Command-line interface for the ML Provenance Framework.
"""

import os
import sys
import logging
import argparse
import yaml
import json
from typing import Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Output to console
        logging.FileHandler('ml_pipeline.log')  # Output to file
    ]
)

from src.tof_ml.pipeline.orchestrator import PipelineOrchestrator
from src.tof_ml.data.data_provenance import ProvenanceTracker
from src.tof_ml.database.api import DBApi


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load the pipeline configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Add the class mapping path if not specified
    if "class_mapping_path" not in config:
        config["class_mapping_path"] = "config/class_mapping_config.yaml"

    return config


def run_pipeline(config: Dict[str, Any], mode: str = "full"):
    """
    Run the ML pipeline with the specified configuration.

    Args:
        config: Pipeline configuration dictionary
        mode: Pipeline execution mode (full, training, evaluation, report)
    """
    # Initialize pipeline orchestrator
    orchestrator = PipelineOrchestrator(config)

    # Run in the specified mode
    if mode == "full":
        metadata = orchestrator.run_pipeline()
    elif mode == "training":
        metadata = orchestrator.run_training()
    elif mode == "evaluation":
        metadata = orchestrator.run_evaluation()
    elif mode == "report":
        metadata = orchestrator.generate_report()
    else:
        raise ValueError(f"Invalid mode: {mode}")

    return metadata


# Database command handlers

def handle_db_command(args):
    """Handle database commands."""
    if not args.db_command:
        print("Error: No database command specified. Use --help for options.")
        sys.exit(1)

    # Initialize database API
    db_api = DBApi(config_path=args.config)

    try:
        if args.db_command == "list-experiments":
            list_experiments(db_api, args)
        elif args.db_command == "show-experiment":
            show_experiment(db_api, args)
        elif args.db_command == "compare":
            compare_experiments(db_api, args)
        elif args.db_command == "query":
            query_experiments(db_api, args)
        elif args.db_command == "export":
            export_experiment(db_api, args)
        elif args.db_command == "delete":
            delete_experiment(db_api, args)
        else:
            print(f"Unknown database command: {args.db_command}")
            sys.exit(1)
    except Exception as e:
        logging.error(f"Database command failed: {e}", exc_info=True)
        sys.exit(1)


def list_experiments(db_api, args):
    """List experiments in the database."""
    experiments = db_api.query_experiments(
        filters={},
        sort_by=args.sort_by,
        limit=args.limit
    )

    if not experiments:
        print("No experiments found in the database.")
        return

    # Print table header
    print(f"{'ID':<24} {'Name':<20} {'Model Type':<15} {'Accuracy':<10} {'Loss':<10} {'Date':<20}")
    print('-' * 100)

    # Print each experiment
    for exp in experiments:
        exp_id = exp.get("id", "N/A")
        name = exp.get("name", "N/A")

        # Get metadata
        try:
            metadata = json.loads(exp.get("metadata", "{}"))
            model_type = metadata.get("model_type", "N/A")

            # Get metrics
            metrics = {}
            if "metric_value" in exp:
                metrics[args.sort_by] = exp["metric_value"]

            accuracy = metrics.get("accuracy", metadata.get("test_metrics", {}).get("accuracy", "N/A"))
            test_loss = metrics.get("test_loss", metadata.get("test_loss", "N/A"))
            date = exp.get("timestamp", "N/A")

            # Format values
            if isinstance(accuracy, float):
                accuracy = f"{accuracy:.4f}"
            if isinstance(test_loss, float):
                test_loss = f"{test_loss:.4f}"

            print(f"{exp_id:<24} {name:<20} {model_type:<15} {accuracy:<10} {test_loss:<10} {date:<20}")
        except json.JSONDecodeError:
            # Fallback if metadata isn't valid JSON
            print(f"{exp_id:<24} {name:<20} {'N/A':<15} {'N/A':<10} {'N/A':<10} {exp.get('timestamp', 'N/A'):<20}")


def show_experiment(db_api, args):
    """Show details of a specific experiment."""
    experiment = db_api.get_experiment(args.experiment_id)

    if not experiment:
        print(f"Experiment '{args.experiment_id}' not found.")
        return

    # Load metadata
    metadata = experiment.get("metadata", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}

    # Load config
    config = experiment.get("config", {})
    if isinstance(config, str):
        try:
            config = json.loads(config)
        except json.JSONDecodeError:
            config = {}

    # Print experiment details
    print(f"Experiment: {experiment.get('name', 'N/A')}")
    print(f"ID: {experiment.get('id', 'N/A')}")
    print(f"Date: {experiment.get('timestamp', 'N/A')}")

    # Print metadata
    print("\nMetadata:")
    for key, value in metadata.items():
        if key not in ["config", "metrics"] and not isinstance(value, dict):
            print(f"  {key}: {value}")

    # Print metrics
    print("\nMetrics:")
    for name, value in experiment.get("metrics", {}).items():
        print(f"  {name}: {value if not isinstance(value, float) else f'{value:.6f}'}")

    # Print model information
    if "model" in experiment:
        print("\nModel:")
        model = experiment["model"]
        print(f"  Path: {model.get('path', 'N/A')}")

    # Print artifacts
    if "artifacts" in experiment:
        print("\nArtifacts:")
        for type_name, path in experiment["artifacts"].items():
            print(f"  {type_name}: {path}")

    # Print configuration summary
    print("\nConfiguration Summary:")
    if "model" in config:
        print("  Model Configuration:")
        for key, value in config["model"].items():
            print(f"    {key}: {value}")

    if "data" in config:
        print("  Data Configuration:")
        for key, value in config["data"].items():
            if not isinstance(value, dict):
                print(f"    {key}: {value}")


def compare_experiments(db_api, args):
    """Compare two or more experiments."""
    # Get comparison data
    comparison = db_api.compare_experiments(args.experiment_ids)

    if not comparison or not comparison.get("experiments"):
        print("No valid experiments to compare.")
        return

    experiments = comparison.get("experiments", [])
    metrics = comparison.get("metrics", {})

    # Print experiment summary
    print("Experiments:")
    for exp in experiments:
        print(f"  - {exp.get('name', 'N/A')} (ID: {exp.get('id', 'N/A')}, Date: {exp.get('timestamp', 'N/A')})")

    print("\nMetrics Comparison:")

    # Filter metrics if specified
    if args.metrics:
        metrics = {k: v for k, v in metrics.items() if k in args.metrics}

    if not metrics:
        print("  No comparable metrics found.")
        return

    # Print table header
    header = "Metric"
    headers = [header]
    for exp in experiments:
        exp_name = exp.get("name", "Unknown")
        exp_id = exp.get("id", "Unknown")
        headers.append(f"{exp_name} ({exp_id[:8]})")

    # Print header
    col_width = 20
    print(" | ".join(h.ljust(col_width) for h in headers))
    print("-" * ((col_width + 3) * len(headers) - 1))

    # Print each metric
    for metric_name, metric_info in metrics.items():
        row = [metric_name.ljust(col_width)]
        metric_values = metric_info.get("values", {})

        for exp in experiments:
            exp_id = exp.get("id", "Unknown")
            value = metric_values.get(exp_id, "N/A")

            if isinstance(value, float):
                value = f"{value:.6f}"

            row.append(str(value).ljust(col_width))

        print(" | ".join(row))


def query_experiments(db_api, args):
    """Query experiments by criteria."""
    # Build query filters
    filters = {}

    if args.experiment_name:
        filters["name"] = args.experiment_name

    if args.model_type:
        filters["model_type"] = args.model_type

    # Execute query
    experiments = db_api.query_experiments(
        filters=filters,
        limit=args.limit
    )

    # Filter results by min_accuracy and max_loss if specified
    # (These may need to be handled after query since SQLite doesn't support complex JSON filtering)
    if args.min_accuracy or args.max_loss:
        filtered_experiments = []
        for exp in experiments:
            metadata = json.loads(exp.get("metadata", "{}"))
            metrics = metadata.get("test_metrics", {})

            accuracy = metrics.get("accuracy", 0.0)
            loss = metadata.get("test_loss", float('inf'))

            if (args.min_accuracy is None or accuracy >= args.min_accuracy) and \
                    (args.max_loss is None or loss <= args.max_loss):
                filtered_experiments.append(exp)

        experiments = filtered_experiments

    if not experiments:
        print("No experiments found matching the query criteria.")
        return

    # Display results
    print(f"Found {len(experiments)} experiments matching criteria:")

    # Reuse list_experiments to display
    args.sort_by = "timestamp"  # Default sort for query results
    list_experiments(db_api, args)


def export_experiment(db_api, args):
    """Export experiment data."""
    experiment = db_api.get_experiment(args.experiment_id)

    if not experiment:
        print(f"Experiment '{args.experiment_id}' not found.")
        return

    # Determine output path
    output_path = args.output if args.output else f"experiment_{args.experiment_id}.{args.format}"

    # Export in specified format
    if args.format == "json":
        with open(output_path, 'w') as f:
            json.dump(experiment, f, indent=2, default=str)
    elif args.format == "yaml":
        with open(output_path, 'w') as f:
            yaml.dump(experiment, f, default_str=str)
    elif args.format == "csv":
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(["Property", "Value"])
            # Write experiment data
            for key, value in _flatten_dict(experiment).items():
                writer.writerow([key, str(value)])

    print(f"Experiment exported to {output_path}")


def delete_experiment(db_api, args):
    """Delete an experiment from the database."""
    # First check if experiment exists
    experiment = db_api.get_experiment(args.experiment_id)

    if not experiment:
        print(f"Experiment '{args.experiment_id}' not found.")
        return

    # Confirm deletion unless --force is used
    if not args.force:
        confirm = input(f"Are you sure you want to delete experiment '{args.experiment_id}'? [y/N] ")
        if confirm.lower() not in ['y', 'yes']:
            print("Deletion cancelled.")
            return

    # Perform deletion
    # Note: We need to implement delete_experiment in DBApi
    try:
        # Create a cursor for deletion
        cursor = db_api.connection.cursor()

        # Delete related records first
        cursor.execute("DELETE FROM metrics WHERE experiment_id = ?", (args.experiment_id,))
        cursor.execute("DELETE FROM models WHERE experiment_id = ?", (args.experiment_id,))
        cursor.execute("DELETE FROM artifacts WHERE experiment_id = ?", (args.experiment_id,))

        # Delete the experiment record
        cursor.execute("DELETE FROM experiments WHERE id = ?", (args.experiment_id,))

        # Commit changes
        db_api.connection.commit()

        print(f"Experiment '{args.experiment_id}' deleted successfully.")
    except Exception as e:
        db_api.connection.rollback()
        print(f"Error deleting experiment: {e}")


def _flatten_dict(d, parent_key='', sep='.'):
    """Flatten a nested dictionary for CSV export."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def main():
    """Main entry point for ML pipeline command."""
    # Create main parser
    parser = argparse.ArgumentParser(description="ML Provenance Pipeline Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the ML pipeline")
    pipeline_parser.add_argument("--config", "-c", type=str, default="config/mnist_config.yaml",
                                 help="Path to the pipeline configuration file")
    pipeline_parser.add_argument("--mode", "-m", type=str, default="full",
                                 choices=["full", "training", "evaluation", "report"],
                                 help="Pipeline execution mode")

    # Provenance management commands - add subparser for experiment mgmt
    prov_parser = subparsers.add_parser("provenance", help="Experiment and provenance management")
    prov_subparsers = prov_parser.add_subparsers(dest="prov_command", help="Provenance command")

    # List experiments
    list_parser = prov_subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--output-dir", dest="output_dir", default="./output",
                             help="Base output directory")

    # List runs for an experiment
    runs_parser = prov_subparsers.add_parser("runs", help="List runs for an experiment")
    runs_parser.add_argument("experiment", help="Experiment name")
    runs_parser.add_argument("--output-dir", dest="output_dir", default="./output",
                             help="Base output directory")

    # Show run details
    show_parser = prov_subparsers.add_parser("show", help="Show run details")
    show_parser.add_argument("experiment", help="Experiment name")
    show_parser.add_argument("run", help="Run ID")
    show_parser.add_argument("--output-dir", dest="output_dir", default="./output",
                             help="Base output directory")

    # Find runs with completed stages
    find_parser = prov_subparsers.add_parser("find", help="Find runs with completed stages")
    find_parser.add_argument("experiment", help="Experiment name")
    find_parser.add_argument("stage", help="Stage name")
    find_parser.add_argument("--output-dir", dest="output_dir", default="./output",
                             help="Base output directory")

    # Add database command
    db_parser = subparsers.add_parser("db", help="Database operations")
    db_subparsers = db_parser.add_subparsers(dest="db_command", help="Database command")

    # List experiments in database
    list_exp_parser = db_subparsers.add_parser("list-experiments", help="List all experiments in the database")
    list_exp_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml",
                                 help="Path to the database configuration file")
    list_exp_parser.add_argument("--limit", "-l", type=int, default=10,
                                 help="Maximum number of experiments to show")
    list_exp_parser.add_argument("--sort-by", "-s", type=str, default="experiment_id",
                                 help="Field to sort by")
    list_exp_parser.add_argument("--desc", action="store_true",
                                 help="Sort in descending order")

    # Show experiment details
    show_exp_parser = db_subparsers.add_parser("show-experiment", help="Show details of a specific experiment")
    show_exp_parser.add_argument("experiment_id", help="ID of the experiment to show")
    show_exp_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml",
                                 help="Path to the database configuration file")

    # Compare experiments
    compare_parser = db_subparsers.add_parser("compare", help="Compare two or more experiments")
    compare_parser.add_argument("experiment_ids", nargs="+", help="IDs of experiments to compare")
    compare_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml",
                                help="Path to the database configuration file")
    compare_parser.add_argument("--metrics", "-m", nargs="+", default=["test_loss", "accuracy"],
                                help="Metrics to compare")

    # Query experiments
    query_parser = db_subparsers.add_parser("query", help="Query experiments by criteria")
    query_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml",
                              help="Path to the database configuration file")
    query_parser.add_argument("--experiment-name", type=str, help="Filter by experiment name")
    query_parser.add_argument("--model-type", type=str, help="Filter by model type")
    query_parser.add_argument("--min-accuracy", type=float, help="Minimum accuracy")
    query_parser.add_argument("--max-loss", type=float, help="Maximum loss value")
    query_parser.add_argument("--limit", "-l", type=int, default=10, help="Maximum results to show")

    # Export experiment data
    export_parser = db_subparsers.add_parser("export", help="Export experiment data")
    export_parser.add_argument("experiment_id", help="ID of the experiment to export")
    export_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml",
                               help="Path to the database configuration file")
    export_parser.add_argument("--format", "-f", choices=["json", "csv", "yaml"], default="json",
                               help="Export format")
    export_parser.add_argument("--output", "-o", type=str, help="Output file path")

    # Delete experiment
    delete_parser = db_subparsers.add_parser("delete", help="Delete an experiment from the database")
    delete_parser.add_argument("experiment_id", help="ID of the experiment to delete")
    delete_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml",
                               help="Path to the database configuration file")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    # Parse arguments
    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "pipeline":
        try:
            # Load configuration
            config = load_config(args.config)

            # Run pipeline
            metadata = run_pipeline(config, args.mode)

            print(f"Pipeline completed successfully. Output directory: {metadata.get('output_dir')}")

        except Exception as e:
            logging.error(f"Pipeline execution failed: {e}", exc_info=True)
            sys.exit(1)

    elif args.command == "provenance":
        # Handle provenance commands using the static CLI helpers
        if args.prov_command == "list":
            experiments = ProvenanceTracker.list_experiments(args.output_dir)
            print("Available experiments:")
            for exp in experiments:
                print(f"  - {exp}")

        elif args.prov_command == "runs":
            tracker = ProvenanceTracker({"experiment_name": args.experiment})
            tracker.experiment_dir = os.path.join(args.output_dir, args.experiment)
            tracker.provenance_db_path = os.path.join(tracker.experiment_dir, "provenance")

            runs = tracker.get_experiment_runs()

            print(f"Runs for experiment '{args.experiment}':")
            for run in runs:
                run_id = run.get("run_id", "unknown")
                start_time = run.get("start_time", "unknown")
                stages = run.get("stages", {})
                completed_stages = [s for s, details in stages.items() if details.get("completed", False)]

                print(f"  - {run_id} (Started: {start_time})")
                print(f"    Completed stages: {', '.join(completed_stages) if completed_stages else 'none'}")

        elif args.prov_command == "show":
            tracker = ProvenanceTracker({"experiment_name": args.experiment})
            tracker.experiment_dir = os.path.join(args.output_dir, args.experiment)
            tracker.provenance_db_path = os.path.join(tracker.experiment_dir, "provenance")

            run_record_path = os.path.join(tracker.provenance_db_path, "runs", f"{args.run}.json")
            if not os.path.exists(run_record_path):
                print(f"Run '{args.run}' not found in experiment '{args.experiment}'")
                return

            with open(run_record_path, 'r') as f:
                run = json.load(f)

            print(f"Details for run '{args.run}' in experiment '{args.experiment}':")
            print(f"  Started: {run.get('start_time', 'unknown')}")
            print(f"  Pipeline hash: {run.get('pipeline_hash', 'unknown')}")

            if "stages" in run:
                print("  Stages:")
                for stage, details in run["stages"].items():
                    status = "Completed" if details.get("completed", False) else "Failed"
                    timestamp = details.get("timestamp", "unknown")
                    print(f"    - {stage}: {status} ({timestamp})")

                    if details.get("artifacts"):
                        print("      Artifacts:")
                        for name, path in details["artifacts"].items():
                            print(f"        - {name}: {path}")

        elif args.prov_command == "find":
            tracker = ProvenanceTracker({"experiment_name": args.experiment})
            tracker.experiment_dir = os.path.join(args.output_dir, args.experiment)
            tracker.provenance_db_path = os.path.join(tracker.experiment_dir, "provenance")

            best_run = tracker.find_best_run_for_stage(args.stage)
            if not best_run:
                print(f"No runs found in experiment '{args.experiment}' with completed stage '{args.stage}'")
                return

            run_id = best_run.get("run_id", "unknown")
            print(f"Best run for stage '{args.stage}' in experiment '{args.experiment}':")
            print(f"  Run ID: {run_id}")
            print(f"  Started: {best_run.get('start_time', 'unknown')}")

            stage_details = best_run.get("stages", {}).get(args.stage, {})
            print(f"  Completed: {stage_details.get('timestamp', 'unknown')}")

            if stage_details.get("artifacts"):
                print("  Artifacts:")
                for name, path in stage_details["artifacts"].items():
                    print(f"    - {name}: {path}")

    elif args.command == "db":
        # Handle database commands
        handle_db_command(args)

    else:
        if not args.command:
            parser.print_help()
        else:
            print(f"Unknown command: {args.command}")
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()