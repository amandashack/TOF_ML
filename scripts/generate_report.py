
import logging
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import subprocess
import sys
import yaml
import json
from notion_client import Client
from notion_client.errors import APIResponseError
from src.tof_ml.logging.logging_utils import setup_logger
from plugins.data_filtering import filter_data
from src.tof_ml.database.drive_utils import upload_file_to_drive, make_file_public, get_public_link

NOTION_API_URL = "https://api.notion.com/v1/pages"
NOTION_VERSION = "2022-06-28"  # or stable version date


def get_range_or_full(data_column, user_range):
    if user_range is not None:
        return user_range
    return (float(np.min(data_column)), float(np.max(data_column)))


def midpoint_of_range(rng):
    return (rng[0] + rng[1]) / 2.0


def load_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def merge_configs(base_config: dict, cli_args: dict) -> dict:
    """
    Merge CLI arguments into the base configuration.
    CLI arguments override values in the base configuration.
    """
    for key, value in cli_args.items():
        if value is not None:
            parts = key.split(".")
            target = base_config
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
    return base_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Report")
    parser.add_argument("--loader_type", type=str, default="nm_csv",
                        choices=["nm_csv", "h5"],
                        help="Which data loader to use: 'nm_csv' or 'h5'")
    parser.add_argument("--data_dir", type=str,
                        help="Path to the directory containing your files.")

    # New arguments
    parser.add_argument("--h5_file", type=str, default=None,
                        help="If provided, load only this single .h5 file (full path) for the H5 loader.")
    parser.add_argument("--parse_data_directories", action="store_true",
                        help="If passed, parse subdirectories named R(\\d+) within data_dir for .h5 files.")
    parser.add_argument("--n_samples", type=int, default=5000, help="Number of samples to plot.")
    parser.add_argument("--mid1_ratio", type=float, nargs=2, default=None, help="Min and max range for mid1_ratio.")
    parser.add_argument("--mid2_ratio", type=float, nargs=2, default=None, help="Min and max range for mid2_ratio.")
    parser.add_argument("--retardation", type=float, nargs=2, default=None, help="Min and max range for retardation.")
    parser.add_argument("--pass_energy", action="store_true", help="Plot pass energy (initial_ke + retardation).")
    parser.add_argument("--filepath", type=str, default=None,
                        help="If provided, save plots to this file path, else show them.")
    parser.add_argument("--notion", action="store_true",
                        help="If provided, push a record to Notion with given data.")
    parser.add_argument("--upload_drive", action="store_true",
                        help="Upload generated plot PNGs to Google Drive and use their public links.")
    parser.add_argument("--drive_folder_id", type=str, default=None,
                        help="Google Drive folder ID to upload plots into.")
    args = parser.parse_args()

    # Set up logging
    setup_logger("data_loader")
    logger = logging.getLogger("data_loader")

    # 1. Load base config
    base_config = load_config("config/data_report_config.yaml")

    # 2. loader_config
    loader_config = load_config("config/class_mapping_config.yaml")

    # 3. database_config
    database_config = load_config("config/database_config.yaml")

    # Merge CLI arguments into config
    if args.data_dir:
        base_config["data"]["directory"] = args.data_dir
    if args.filepath:
        base_config["plotting"]["filepath"] = args.filepath
    if args.notion:
        base_config["notion"]["enabled"] = args.notion
    if args.upload_drive:
        base_config["google_drive"]["upload_drive"] = args.upload_drive

    # ------------------------------------------------------------------------
    # PICK LOADER AND LOAD DATA
    # ------------------------------------------------------------------------
    # Determine loader
    loader_key = base_config["data"]["loader_config_key"]
    loader_class_path = loader_config[loader_key]["loader_class"]
    module_name, class_name = loader_class_path.rsplit(".", 1)
    loader_module = __import__(module_name, fromlist=[class_name])
    LoaderClass = getattr(loader_module, class_name)

    # Initialize loader
    loader = LoaderClass(config=base_config["data"])
    logger.info("Loading data...")
    data = loader.load_data()
    if data.size == 0:
        logger.error("No data loaded. Exiting.")
        sys.exit(1)

    # Columns:
    # 0: initial_ke
    # 1: initial_elevation
    # 2: mid1_ratio
    # 3: mid2_ratio
    # 4: retardation
    # 5: tof_values
    # 6: x_tof
    # 7: y_tof

    # APPLY FILTERS
    logger.info("Applying Data Filters...")
    loader_params = base_config['data']['parameters']
    mid1 = loader_params.get("mid1", None)
    if mid1 is None:
        mid1 = [float(np.min(data[:, 2])), float(np.max(data[:, 2]))]
    mid2 = loader_params.get("mid2", None)
    if mid2 is None:
        mid2 = [float(np.min(data[:, 3])), float(np.max(data[:, 3]))]
    retardation_range = loader_params.get("retardation_range", None)
    if retardation_range is None:
        retardation_range = [float(np.min(data[:, 4])), float(np.max(data[:, 4]))]
    number_of_samples = loader_params.get("number_of_samples", None)
    df_filtered = filter_data(
        data,
        retardation_range=None if retardation_range == "none" else retardation_range,
        mid1=None if mid1 == "none" else mid1,
        mid2=None if mid2 == "none" else mid2,
        number_of_samples=number_of_samples if number_of_samples else None,
        random_state=42
    )

    if data.size == 0:
        logger.warning("After filtering, no data remains.")
        sys.exit(0)

    n_samples = loader_params['number_of_samples']
    if n_samples is not None and n_samples < len(data):
        indices = np.random.choice(len(data), n_samples, replace=False)
        data = data[indices, :]

    # Decide which energy to plot
    if base_config['plotting']['pass_energy']:
        energy_to_plot = data[:, 0] + data[:, 4]  # initial_ke + retardation
        energy_label = "Pass Energy"
    else:
        energy_to_plot = data[:, 0]  # initial_ke
        energy_label = "Initial Kinetic Energy"

    tof = data[:, 5]
    retardation_vals = data[:, 4]
    x_tof = data[:, 6]

    # ------------------------------------------------------
    # PLOT 1: Before masking (x_tof > 406 not removed)
    # ------------------------------------------------------
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(energy_to_plot, tof, c=retardation_vals, cmap='viridis', s=5, alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylabel('Time of Flight (log scale)')
    plt.xlabel(f'{energy_label} (log scale)')
    plt.title(f'{energy_label} vs. Time of Flight (Before Masking)')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Retardation')

    if base_config['plotting']['filepath']:
        os.makedirs(base_config['plotting']['filepath'], exist_ok=True)
        plot1_path = os.path.join(base_config['plotting']['filepath'], "ke_vs_tof_before_masking.png")
        plt.savefig(plot1_path)
        plt.close()
    else:
        plt.show()
        plot1_path = None

    # ------------------------------------------------------
    # PLOT 2: After masking x_tof >= 406
    # ------------------------------------------------------
    masked_data = data[data[:, 6] >= 406]
    if masked_data.size > 0:
        tof_masked = masked_data[:, 5]
        ret_masked = masked_data[:, 4]
        if args.pass_energy:
            energy_masked = masked_data[:, 0] + masked_data[:, 4]
        else:
            energy_masked = masked_data[:, 0]

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(energy_masked, tof_masked, c=ret_masked, cmap='viridis', s=5, alpha=0.7)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(f'{energy_label} (log scale)')
        plt.ylabel('Time of Flight (log scale)')
        plt.title(f'{energy_label} vs. Time of Flight (After Masking)')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Retardation')

        if base_config['plotting']['filepath']:
            plot2_path = os.path.join(base_config['plotting']['filepath'], "ke_vs_tof_after_masking.png")
            plt.savefig(plot2_path)
            plt.close()
        else:
            plt.show()
            plot2_path = None
    else:
        logger.warning("No data after applying x_tof >= 406 mask.")
        plot2_path = None

    logger.info("Plots generated successfully.")

    # ------------------------------------------------------
    # OPTIONAL: UPLOAD TO GOOGLE DRIVE
    # ------------------------------------------------------
    plot1_url = None
    plot2_url = None
    if base_config['google_drive']['upload_drive']:
        folder_id = database_config['google_drive']['folder_id']

        # Attempt uploading first plot
        if plot1_path:
            file_id = upload_file_to_drive(plot1_path, folder_id=folder_id)
            if file_id:
                make_file_public(file_id)
                plot1_url = get_public_link(file_id)

        # Attempt uploading second plot
        if plot2_path:
            file_id = upload_file_to_drive(plot2_path, folder_id=folder_id)
            if file_id:
                make_file_public(file_id)
                plot2_url = get_public_link(file_id)

    else:
        # If not uploading, provide local file paths or placeholders
        logger.warning("No files were uploaded to google drive")

    # ------------------------------------------------------
    # NOTION INTEGRATION
    # ------------------------------------------------------
    if base_config['notion']['enabled']:
        notion_section = database_config.get("notion", {})
        token_file = notion_section.get("token_file", "notion_token.json")
        notion_db_id = notion_section.get("database_id")

        if not os.path.exists(token_file):
            raise FileNotFoundError(f"Notion token file not found: {token_file}")

        with open(token_file, "r") as f:
            data = json.load(f)
        notion_token = data.get("token")
        if not notion_token:
            raise ValueError("No 'token' found in notion_token.json.")

        notion_client = Client(auth=notion_token)

        current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
        except Exception:
            commit_id = "unknown_commit"

        fpath = base_config['plotting']['filepath']
        file_type = loader_config[loader_key]['loader_type']
        loader_name = LoaderClass.__name__

        payload = {
            "date": {
                "date": {
                    "start": current_date  # Ensure ISO 8601 format
                }
            },
            "git commit ID": {
                "rich_text": [
                    {
                        "text": {"content": str(commit_id)}
                    }
                ]
            },
            "file path": {
                "title": [
                    {
                        "text": {"content": str(fpath)}
                    }
                ]
            },
            "file type": {
                "rich_text": [
                    {
                        "text": {"content": str(file_type)}
                    }
                ]
            },
            "loader": {
                "rich_text": [
                    {
                        "text": {"content": str(loader_name)}
                    }
                ]
            },
            "mid1 ratio": {
                "multi_select": [{"name": str(value)} for value in mid1]
            },
            "mid2 ratio": {
                "multi_select": [{"name": str(value)} for value in mid2]
            },
            "retardation": {
                "multi_select": [{"name": str(value)} for value in retardation_range]
            },
            "plot1": {
                "files": [
                    {
                        "type": "external",
                        "name": "plot1",
                        "external": {"url": str(plot1_url)}
                    }
                ]
            },
            "plot2": {
                "files": [
                    {
                        "type": "external",
                        "name": "plot2",
                        "external": {"url": str(plot2_url)}
                    }
                ]
            }
        }

        # Step 4: Create page in Notion
        try:
            new_page = notion_client.pages.create(
                parent={"database_id": notion_db_id},
                properties=payload
            )
            logger.info(f"Created new Notion page: {new_page['id']}")
        except APIResponseError as e:
            logger.error(f"Failed to create new page: {e}")
            raise
