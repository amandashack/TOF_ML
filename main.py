#!/usr/bin/env python3
"""
Enhanced Command-line interface for the ML Provenance Framework.
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
        logging.StreamHandler(),
        logging.FileHandler('ml_pipeline.log')
    ]
)

logger = logging.getLogger(__name__)

from src.tof_ml.pipeline.orchestrator import PipelineOrchestrator
from src.tof_ml.database.api import DBApi


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the pipeline configuration from a YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Add the class mapping path if not specified
    if "class_mapping_path" not in config:
        config["class_mapping_path"] = "config/class_mapping_config.yaml"

    # Add database config if not specified
    if "database" not in config:
        config["database"] = {"config_path": "config/database_config.yaml"}

    # Add config path for reference
    config["config_path"] = config_path

    return config


def run_pipeline(config: Dict[str, Any], mode: str = "full"):
    """Run the ML pipeline with the specified configuration."""
    try:
        # Initialize enhanced pipeline orchestrator
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

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


def handle_db_command(args):
    """Handle database commands with enhanced functionality."""
    if not args.db_command:
        print("Error: No database command specified. Use --help for options.")
        sys.exit(1)

    logger.info("Processing database request...")

    # Initialize database API
    db_api = DBApi(config_path=args.config)

    try:
        if args.db_command == "list-experiments":
            list_experiments(db_api, args)
        elif args.db_command == "list-runs":
            list_runs(db_api, args)
        elif args.db_command == "show-run":
            show_run(db_api, args)
        elif args.db_command == "compare-runs":
            compare_runs(db_api, args)
        elif args.db_command == "show-lineage":
            show_lineage(db_api, args)
        elif args.db_command == "query-artifacts":
            query_artifacts(db_api, args)
        elif args.db_command == "export-run":
            export_run(db_api, args)
        elif args.db_command == "delete-run":
            delete_run(db_api, args)
        elif args.db_command == "summary":
            show_summary(db_api, args)
        elif args.db_command == "monitor":
            monitor_training(db_api, args)
        else:
            print(f"Unknown database command: {args.db_command}")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Database command failed: {e}", exc_info=True)
        sys.exit(1)


def list_experiments(db_api, args):
    """List all experiments."""
    experiments = db_api.get_experiments(limit=args.limit)

    if not experiments:
        print("No experiments found in the database.")
        return

    # Print table header
    print(f"{'ID':<30} {'Name':<25} {'Created':<20} {'Git SHA':<12} {'Runs':<8} {'Completed':<10}")
    print('-' * 110)

    # Print each experiment
    for exp in experiments:
        exp_id = exp["id"][:29] if len(exp["id"]) > 29 else exp["id"]
        name = exp["name"][:24] if len(exp["name"]) > 24 else exp["name"]
        created = exp["created_at"][:19] if exp["created_at"] else "N/A"
        git_sha = exp["git_commit_sha"][:11] if exp["git_commit_sha"] else "N/A"
        total_runs = exp["total_runs"] or 0
        completed_runs = exp["completed_runs"] or 0

        print(f"{exp_id:<30} {name:<25} {created:<20} {git_sha:<12} {total_runs:<8} {completed_runs:<10}")


def list_runs(db_api, args):
    """List runs for an experiment."""
    # Find experiment using abstracted method
    experiment = db_api.find_experiment_by_name_or_id(args.experiment)
    if not experiment:
        print(f"Experiment '{args.experiment}' not found.")
        return

    experiment_id = experiment["id"]
    runs = db_api.get_runs_for_experiment(experiment_id, limit=args.limit)

    if not runs:
        print(f"No runs found for experiment '{args.experiment}'.")
        return

    # Print table header
    print(f"{'Run ID':<30} {'Created':<20} {'Status':<12} {'Data Reused':<12} {'Test Loss':<12}")
    print('-' * 90)

    # Print each run
    for run in runs:
        run_id = run["id"][:29] if len(run["id"]) > 29 else run["id"]
        created = run["created_at"][:19] if run["created_at"] else "N/A"
        status = run["status"] or "unknown"
        data_reused = "Yes" if run["data_reused"] else "No"
        test_loss = f"{run.get('avg_test_loss', 0):.4f}" if run.get('avg_test_loss') else "N/A"

        print(f"{run_id:<30} {created:<20} {status:<12} {data_reused:<12} {test_loss:<12}")


def show_run(db_api, args):
    """Show detailed information about a specific run."""
    lineage = db_api.get_run_lineage(args.run_id)

    if not lineage or not lineage.get("run_info"):
        print(f"Run '{args.run_id}' not found.")
        return

    run_info = lineage["run_info"]

    print(f"Run: {run_info.get('id', 'N/A')}")
    print(f"Experiment: {run_info.get('experiment_id', 'N/A')}")
    print(f"Status: {run_info.get('status', 'N/A')}")
    print(f"Created: {run_info.get('created_at', 'N/A')}")
    print(f"Completed: {run_info.get('completed_at', 'N/A')}")
    print(f"Data Reused: {'Yes' if run_info.get('data_reused') else 'No'}")

    # Show artifacts
    artifacts = lineage.get("artifacts", [])
    if artifacts:
        print(f"\nArtifacts ({len(artifacts)}):")
        for artifact in artifacts:
            print(f"  - {artifact['artifact_role']} ({artifact['stage']}): {artifact['artifact_type']}")
            print(f"    Path: {artifact['file_path']} {'✓' if os.path.exists(artifact['file_path']) else '✗'}")
            print(f"    Size: {artifact.get('size_bytes', 0)} bytes")

    # Show metrics by stage
    metrics = lineage.get("metrics", [])
    if metrics:
        print(f"\nMetrics ({len(metrics)}):")
        metrics_by_stage = {}
        for metric in metrics:
            stage = metric.get('stage', 'unknown')
            if stage not in metrics_by_stage:
                metrics_by_stage[stage] = []
            metrics_by_stage[stage].append(metric)

        for stage, stage_metrics in metrics_by_stage.items():
            print(f"  {stage.upper()}:")
            for metric in sorted(stage_metrics, key=lambda x: x.get('epoch', 0)):
                epoch_info = f" (epoch {metric['epoch']})" if metric.get('epoch') else ""
                print(f"    - {metric['metric_name']}: {metric['metric_value']:.6f}{epoch_info}")


def monitor_training(db_api, args):
    """Monitor real-time training progress."""
    import time

    print(f"Monitoring training progress for run: {args.run_id}")
    print("Press Ctrl+C to stop monitoring\n")

    try:
        last_epoch = -1
        while True:
            progress = db_api.get_training_progress(args.run_id)

            if progress["status"] == "no_training_data":
                print("No training data found for this run.")
                break

            current_epoch = progress.get("current_epoch", 0)
            if current_epoch > last_epoch:
                latest_metrics = progress.get("latest_metrics", {})
                print(f"Epoch {current_epoch}: " +
                      ", ".join([f"{k}={v:.4f}" for k, v in latest_metrics.items()]))
                last_epoch = current_epoch

            # Check if training is complete using abstracted method
            status = db_api.query_service.get_run_status(args.run_id)
            if status == "completed":
                print(f"\nTraining completed! Final metrics:")
                latest_metrics = progress.get("latest_metrics", {})
                for k, v in latest_metrics.items():
                    print(f"  {k}: {v:.4f}")
                break

            time.sleep(5)  # Check every 5 seconds

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def compare_runs(db_api, args):
    """Compare multiple runs."""
    comparison = db_api.compare_runs(args.run_ids)

    if not comparison or not comparison.get("runs"):
        print("No valid runs to compare.")
        return

    runs = comparison.get("runs", [])
    metrics_comparison = comparison.get("metrics_comparison", {})

    print("Run Comparison:")
    print(f"{'Run ID':<30} {'Experiment':<20} {'Status':<12} {'Created':<20}")
    print('-' * 85)

    for run in runs:
        run_id = run["id"][:29] if len(run["id"]) > 29 else run["id"]
        exp_name = run.get("experiment_name", "N/A")[:19]
        status = run.get("status", "N/A")
        created = run.get("created_at", "N/A")[:19]

        print(f"{run_id:<30} {exp_name:<20} {status:<12} {created:<20}")

    if metrics_comparison:
        print(f"\nMetrics Comparison:")
        header = ["Metric"] + [run["id"][:15] for run in runs]
        col_width = 18
        print(" | ".join(h.ljust(col_width) for h in header))
        print("-" * ((col_width + 3) * len(header) - 1))

        for metric_name, values in metrics_comparison.items():
            row = [metric_name.ljust(col_width)]
            for run in runs:
                run_id = run["id"]
                value = values.get(run_id, "N/A")
                if isinstance(value, float):
                    value_str = f"{value:.6f}"
                else:
                    value_str = str(value)
                row.append(value_str.ljust(col_width))
            print(" | ".join(row))


def show_lineage(db_api, args):
    """Show data lineage for a run."""
    lineage = db_api.get_run_lineage(args.run_id)

    if not lineage:
        print(f"No lineage found for run '{args.run_id}'.")
        return

    transformation_lineage = lineage.get("transformation_lineage", [])

    if not transformation_lineage:
        print(f"No transformation lineage found for run '{args.run_id}'.")
        return

    print(f"Data Lineage for Run: {args.run_id}")
    print(f"{'Stage':<20} {'Transformation':<25} {'Input Hash':<16} {'Output Hash':<16}")
    print('-' * 80)

    for entry in transformation_lineage:
        stage = entry.get("stage", "N/A")
        transformation = entry.get("transformation_name", "N/A")
        input_hash = entry.get("input_hash", "N/A")[:15] if entry.get("input_hash") else "N/A"
        output_hash = entry.get("output_hash", "N/A")[:15]

        print(f"{stage:<20} {transformation:<25} {input_hash:<16} {output_hash:<16}")


def query_artifacts(db_api, args):
    """Query artifacts by type or stage."""
    artifacts = db_api.query_artifacts_by_criteria(
        artifact_type=args.artifact_type,
        stage=args.stage,
        limit=args.limit
    )

    if not artifacts:
        print("No artifacts found matching criteria.")
        return

    print(f"{'Hash ID':<16} {'Type':<10} {'Role':<15} {'Stage':<15} {'Size':<10} {'Created':<20}")
    print('-' * 90)

    for artifact in artifacts:
        hash_id = artifact["hash_id"][:15]
        artifact_type = artifact["artifact_type"]
        role = artifact["artifact_role"][:14]
        stage = artifact["stage"][:14]
        size = f"{artifact['size_bytes']}B" if artifact['size_bytes'] else "N/A"
        created = artifact["created_at"][:19] if artifact["created_at"] else "N/A"

        print(f"{hash_id:<16} {artifact_type:<10} {role:<15} {stage:<15} {size:<10} {created:<20}")


def export_run(db_api, args):
    """Export complete run data."""
    lineage = db_api.get_run_lineage(args.run_id)

    if not lineage:
        print(f"Run '{args.run_id}' not found.")
        return

    output_path = args.output if args.output else f"run_{args.run_id}.{args.format}"

    if args.format == "json":
        with open(output_path, 'w') as f:
            json.dump(lineage, f, indent=2, default=str)
    elif args.format == "yaml":
        import yaml
        with open(output_path, 'w') as f:
            yaml.dump(lineage, f, default_flow_style=False)
    else:
        print(f"Unsupported format: {args.format}")
        return

    print(f"Run data exported to {output_path}")


def delete_run(db_api, args):
    """Delete a run and all associated data."""
    lineage = db_api.get_run_lineage(args.run_id)

    if not lineage or not lineage.get("run_info"):
        print(f"Run '{args.run_id}' not found.")
        return

    if not args.force:
        confirm = input(f"Are you sure you want to delete run '{args.run_id}'? [y/N] ")
        if confirm.lower() not in ['y', 'yes']:
            print("Deletion cancelled.")
            return

    if db_api.delete_run_cascade(args.run_id):
        print(f"Run '{args.run_id}' deleted successfully.")
    else:
        print(f"Error deleting run '{args.run_id}'.")


def show_summary(db_api, args):
    """Show database summary statistics."""
    summary = db_api.get_database_summary()

    print("Database Summary:")
    print("-" * 40)
    print(f"Total Experiments: {summary['experiments']}")
    print(f"Total Runs: {summary['runs']}")
    print(f"Completed Runs: {summary['completed_runs']}")
    print(f"Failed/Running Runs: {summary['failed_runs']}")
    print(f"Runs with Data Reuse: {summary['data_reused_runs']}")
    print(f"Total Artifacts: {summary['artifacts']}")
    print(f"Total Storage: {summary['total_storage_bytes'] / (1024 * 1024):.2f} MB")

    if summary['runs'] > 0:
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Data Reuse Rate: {summary['data_reuse_rate']:.1f}%")


def main():
    """Main entry point for enhanced ML pipeline command."""
    parser = argparse.ArgumentParser(description="Enhanced ML Provenance Pipeline Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run the ML pipeline")
    pipeline_parser.add_argument("--config", "-c", type=str, default="config/mnist_cnn_config.yaml",
                                 help="Path to the pipeline configuration file")
    pipeline_parser.add_argument("--mode", "-m", type=str, default="full",
                                 choices=["full", "training", "evaluation", "report"],
                                 help="Pipeline execution mode")

    # Database commands
    db_parser = subparsers.add_parser("db", help="Database operations")
    db_subparsers = db_parser.add_subparsers(dest="db_command", help="Database command")

    # Add all the database subcommands
    list_exp_parser = db_subparsers.add_parser("list-experiments", help="List all experiments")
    list_exp_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")
    list_exp_parser.add_argument("--limit", "-l", type=int, default=20)

    list_runs_parser = db_subparsers.add_parser("list-runs", help="List runs for an experiment")
    list_runs_parser.add_argument("experiment", help="Experiment name or ID")
    list_runs_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")
    list_runs_parser.add_argument("--limit", "-l", type=int, default=20)

    show_run_parser = db_subparsers.add_parser("show-run", help="Show details of a specific run")
    show_run_parser.add_argument("run_id", help="ID of the run to show")
    show_run_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")

    compare_parser = db_subparsers.add_parser("compare-runs", help="Compare multiple runs")
    compare_parser.add_argument("run_ids", nargs="+", help="IDs of runs to compare")
    compare_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")

    lineage_parser = db_subparsers.add_parser("show-lineage", help="Show data lineage for a run")
    lineage_parser.add_argument("run_id", help="ID of the run")
    lineage_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")

    artifacts_parser = db_subparsers.add_parser("query-artifacts", help="Query artifacts")
    artifacts_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")
    artifacts_parser.add_argument("--artifact-type", type=str, help="Filter by artifact type")
    artifacts_parser.add_argument("--stage", type=str, help="Filter by stage")
    artifacts_parser.add_argument("--limit", "-l", type=int, default=20)

    export_parser = db_subparsers.add_parser("export-run", help="Export run data")
    export_parser.add_argument("run_id", help="ID of the run to export")
    export_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")
    export_parser.add_argument("--format", "-f", choices=["json", "yaml"], default="json")
    export_parser.add_argument("--output", "-o", type=str, help="Output file path")

    delete_parser = db_subparsers.add_parser("delete-run", help="Delete a run")
    delete_parser.add_argument("run_id", help="ID of the run to delete")
    delete_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")
    delete_parser.add_argument("--force", action="store_true", help="Skip confirmation")

    summary_parser = db_subparsers.add_parser("summary", help="Show database summary")
    summary_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")

    # Add training monitoring command
    monitor_parser = db_subparsers.add_parser("monitor", help="Monitor real-time training progress")
    monitor_parser.add_argument("run_id", help="ID of the run to monitor")
    monitor_parser.add_argument("--config", "-c", type=str, default="config/database_config.yaml")

    args = parser.parse_args()

    # Execute the appropriate command
    if args.command == "pipeline":
        try:
            config = load_config(args.config)
            metadata = run_pipeline(config, args.mode)

            print(f"Pipeline completed successfully.")
            print(f"Experiment ID: {metadata.get('experiment_id')}")
            print(f"Run ID: {metadata.get('run_id')}")

        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}", exc_info=True)
            sys.exit(1)

    elif args.command == "db":
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