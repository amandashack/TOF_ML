import logging
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import datetime
import subprocess
import requests

from src.utils.logging_utils import setup_logging
from services.data_loader import load_stacked_data

# NEW: import your google_drive_uploader module
from src.utils.google_drive_uploader import (
    upload_file_to_drive,
    make_file_public,
    get_public_link
)

NOTION_API_URL = "https://api.notion.com/v1/pages"
NOTION_VERSION = "2022-06-28"  # Update to a stable version date if needed

def get_range_or_full(data_column, user_range):
    if user_range is not None:
        return user_range
    return (float(np.min(data_column)), float(np.max(data_column)))

def midpoint_of_range(rng):
    return (rng[0] + rng[1]) / 2.0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Report")

    parser.add_argument("--data_dir", type=str, default=r"C:\Users\proxi\Documents\coding\TOF_data\R0",
                        help="Path to the data directory containing .h5 files.")
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples to plot.")
    parser.add_argument("--mid1_ratio", type=float, nargs=2, default=None, help="Min and max range for mid1_ratio.")
    parser.add_argument("--mid2_ratio", type=float, nargs=2, default=None, help="Min and max range for mid2_ratio.")
    parser.add_argument("--retardation", type=float, nargs=2, default=None, help="Min and max range for retardation.")
    parser.add_argument("--pass_energy", action="store_true", help="Plot pass energy (initial_ke + retardation) instead of initial_ke.")
    parser.add_argument("--filepath", type=str, default=None, help="If provided, save plots to this file path, else show plots.")
    parser.add_argument("--notion", action="store_true", help="If provided, push a record to Notion with given data.")
    # NEW: optionally upload plot images to Drive
    parser.add_argument("--upload_drive", action="store_true",
                        help="If provided, upload generated plot PNGs to Google Drive and use their public links in Notion.")
    parser.add_argument("--drive_folder_id", type=str, default=None,
                        help="Optional Google Drive folder ID to upload plots into.")

    args = parser.parse_args()

    # Set up logging
    setup_logging(log_file="reports/logs/data_loading.log", level=logging.DEBUG)
    logger = logging.getLogger("data_loader")

    # Load data
    data = load_stacked_data(args.data_dir)

    if data.size == 0:
        logger.error("No data loaded. Exiting.")
        exit(1)

    # Columns:
    # 0: initial_ke
    # 1: initial_elevation
    # 2: x_tof
    # 3: y_tof
    # 4: mid1_ratio
    # 5: mid2_ratio
    # 6: retardation
    # 7: tof_values

    mid1_range = get_range_or_full(data[:,4], args.mid1_ratio)
    mid2_range = get_range_or_full(data[:,5], args.mid2_ratio)
    retard_range = get_range_or_full(data[:,6], args.retardation)

    # Apply filters
    data = data[(data[:,4] >= mid1_range[0]) & (data[:,4] <= mid1_range[1])]
    data = data[(data[:,5] >= mid2_range[0]) & (data[:,5] <= mid2_range[1])]
    data = data[(data[:,6] >= retard_range[0]) & (data[:,6] <= retard_range[1])]

    if data.size == 0:
        logger.warning("After filtering, no data remains.")
        exit(0)

    if args.n_samples is not None and args.n_samples < len(data):
        indices = np.random.choice(len(data), args.n_samples, replace=False)
        data = data[indices, :]

    # Compute the energy to plot
    if args.pass_energy:
        energy_to_plot = data[:,0] + data[:,6]  # initial_ke + retardation
        energy_label = "Pass Energy"
    else:
        energy_to_plot = data[:,0]  # initial_ke
        energy_label = "Initial Kinetic Energy"

    tof = data[:,7]
    retardation = data[:,6]
    x_tof = data[:,2]

    # --------------------------------------------------------------
    # PLOT 1: Before masking (x_tof > 406 not removed)
    # --------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(tof, energy_to_plot, c=retardation, cmap='viridis', s=5, alpha=0.7)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Time of Flight (log scale)')
    plt.ylabel(f'{energy_label} (log scale)')
    plt.title(f'{energy_label} vs. Time of Flight (Before Masking)')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Retardation')

    if args.filepath:
        # Ensure the directory exists
        os.makedirs(args.filepath, exist_ok=True)
        plot1_path = os.path.join(args.filepath, "ke_vs_tof_before_masking.png")
        plt.savefig(plot1_path)
        plt.close()
    else:
        plt.show()
        plot1_path = None

    # --------------------------------------------------------------
    # PLOT 2: After masking out values where x_tof > 406
    # --------------------------------------------------------------
    masked_data = data[data[:,2] >= 406]
    if masked_data.size > 0:
        tof_masked = masked_data[:,7]
        retardation_masked = masked_data[:,6]
        if args.pass_energy:
            energy_masked = masked_data[:,0] + masked_data[:,6]
        else:
            energy_masked = masked_data[:,0]

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(tof_masked, energy_masked, c=retardation_masked, cmap='viridis', s=5, alpha=0.7)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Time of Flight (log scale)')
        plt.ylabel(f'{energy_label} (log scale)')
        plt.title(f'{energy_label} vs. Time of Flight (After Masking)')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Retardation')

        if args.filepath:
            plot2_path = os.path.join(args.filepath, "ke_vs_tof_after_masking.png")
            plt.savefig(plot2_path)
            plt.close()
        else:
            plt.show()
            plot2_path = None
    else:
        logger.warning("No data after masking out x_tof > 406.")
        plot2_path = None

    logger.info("Plots generated successfully.")

    # --------------------------------------------------------------
    # Optional: Upload to Google Drive
    # --------------------------------------------------------------
    plot1_url = None
    plot2_url = None
    if args.upload_drive:
        folder_id = args.drive_folder_id  # e.g., '1AbCdEfGhiJKXYZ...' or None

        if plot1_path:
            file_id = upload_file_to_drive(plot1_path, folder_id=folder_id)
            if file_id:
                # Make the file public
                make_file_public(file_id)
                # Get the shareable link
                plot1_url = get_public_link(file_id)

        if plot2_path:
            file_id = upload_file_to_drive(plot2_path, folder_id=folder_id)
            if file_id:
                make_file_public(file_id)
                plot2_url = get_public_link(file_id)

    else:
        # If not uploading, either supply local file paths or dummy links
        if plot1_path:
            plot1_url = f"file://{os.path.abspath(plot1_path)}"
        else:
            plot1_url = "https://example.com/no_plot1.png"

        if plot2_path:
            plot2_url = f"file://{os.path.abspath(plot2_path)}"
        else:
            plot2_url = "https://example.com/no_plot2.png"

    # --------------------------------------------------------------
    # Notion Integration
    # --------------------------------------------------------------
    if args.notion:
        notion_token = "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"
        database_id = "160d57b3c56d808986eed4a41b7e5512"

        current_date = datetime.date.today().isoformat()
        try:
            commit_id = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
        except Exception:
            commit_id = "unknown_commit"

        fpath = args.filepath if args.filepath else "No filepath provided"
        file_type = "h5"
        loader_name = "load_stacked_data"

        mid1_val = midpoint_of_range(mid1_range)
        mid2_val = midpoint_of_range(mid2_range)
        retard_val = midpoint_of_range(retard_range)

        headers = {
            "Authorization": f"Bearer {notion_token}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json"
        }

        payload = {
            "parent": {"database_id": database_id},
            "properties": {
                "date": {
                    "date": {"start": current_date}
                },
                "git commit ID": {
                    "rich_text": [{
                        "text": {"content": commit_id}
                    }]
                },
                "file path": {
                    "title": [{
                        "text": {"content": fpath}
                    }]
                },
                "file type": {
                    "rich_text": [{
                        "text": {"content": file_type}
                    }]
                },
                "loader": {
                    "rich_text": [{
                        "text": {"content": loader_name}
                    }]
                },
                "mid1_ratio": {
                    "number": mid1_val
                },
                "mid2_ratio": {
                    "number": mid2_val
                },
                "retardation": {
                    "number": retard_val
                },
                "plot1": {
                    "files": [{
                        "type": "external",
                        "name": "plot1",
                        "external": {
                            "url": plot1_url
                        }
                    }]
                },
                "plot2": {
                    "files": [{
                        "type": "external",
                        "name": "plot2",
                        "external": {
                            "url": plot2_url
                        }
                    }]
                }
            }
        }

        response = requests.post(NOTION_API_URL, headers=headers, json=payload)
        if response.status_code in (200, 201):
            logger.info("Record successfully inserted into Notion database.")
        else:
            logger.error(
                f"Failed to insert record into Notion. Status: {response.status_code}, Response: {response.text}"
            )



