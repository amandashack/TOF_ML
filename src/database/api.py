import os
import json
import logging
import subprocess
from datetime import datetime
from typing import Dict, Any

from notion_client import Client
from notion_client.errors import APIResponseError
from src.database.drive_utils import (
    upload_file_to_drive,
    make_file_public,
    get_public_link,
)
import yaml

logger = logging.getLogger('trainer')

class DBApi:
    def __init__(self, config_path: str = "config/database_config.yaml"):
        self.config = self._load_config(config_path)

        notion_section = self.config.get("notion", {})
        token_file = notion_section.get("token_file", "notion_token.json")
        self.notion_db_id = notion_section.get("database_id")

        if not os.path.exists(token_file):
            raise FileNotFoundError(f"Notion token file not found: {token_file}")

        with open(token_file, "r") as f:
            data = json.load(f)
        notion_token = data.get("token")
        if not notion_token:
            raise ValueError("No 'token' found in notion_token.json.")

        self.notion_client = Client(auth=notion_token)

        drive_section = self.config.get("google_drive", {})
        self.drive_folder_id = drive_section.get("folder_id")
        self.drive_scopes = drive_section.get("scopes", ["https://www.googleapis.com/auth/drive.file"])
        self.drive_token_file = drive_section.get("token_file", "token.json")
        self.client_secret_file = drive_section.get("client_secret_file")

        # Fetch existing properties once during initialization
        self.existing_properties = self._fetch_existing_properties()

    def _fetch_existing_properties(self) -> Dict[str, Any]:
        try:
            db = self.notion_client.databases.retrieve(self.notion_db_id)
            properties = db.get("properties", {})
            return properties
        except APIResponseError as e:
            logger.error(f"Failed to retrieve database properties: {e}")
            return {}

    def _update_database_schema(self, new_properties: Dict[str, Any]):
        try:
            self.notion_client.databases.update(
                self.notion_db_id,
                properties=new_properties
            )
            logger.info("Database schema updated with new properties.")
            # Update the existing_properties cache
            self.existing_properties.update(new_properties)
        except APIResponseError as e:
            logger.error(f"Failed to update database schema: {e}")

    def _define_property(self, prop_name: str, prop_type: str):
        """
        Define a Notion property based on the desired type.
        Extend this method to handle more property types as needed.
        """
        type_mapping = {
            "title": {
                "title": {}
            },
            "date": {
                "date": {}
            },
            "rich_text": {
                "rich_text": {}
            },
            "number": {
                "number": {
                    "format": "number"
                }
            },
            "multi_select": {
                "multi_select": {
                    "options": []  # Initially empty; can be updated if needed
                }
            },
            "url": {
                "url": {}
            },
            "files": {  # Added 'files' type
                "files": {}
            }
            # Add more mappings as needed
        }
        return type_mapping.get(prop_type, {"rich_text": {}})

    def _ensure_properties_exist(self, desired_properties: Dict[str, str]):
        """
        Ensure that all desired properties exist in the Notion database.
        :param desired_properties: Dict where key is property name and value is property type.
        """
        missing_properties = {}
        for prop_name, prop_type in desired_properties.items():
            if prop_name not in self.existing_properties:
                missing_properties[prop_name] = self._define_property(prop_name, prop_type)

        if missing_properties:
            self._update_database_schema(missing_properties)

    def record_model_run(
        self,
        config_dict: Dict[str, Any],
        training_results: Dict[str, Any],
        model_path: str,
        plot_paths: Dict[str, str]
    ):
        """
        Records a training run to Notion:
          - config_dict: e.g. entire base config
          - training_results: e.g. {"val_mse": ..., "test_mse": ...}
          - model_path: path to the final model
          - plot_paths: e.g. { "true_vs_pred": "...", "residuals": "...", ... }

        Steps:
          1. Ensure all necessary properties exist in the database.
          2. Upload each plot to Drive
          3. Create a new row in Notion with separate columns for each config key
             and each training result key.
        """
        # Step 1: Define desired properties based on config_dict and training_results
        desired_properties = self._gather_desired_properties(config_dict, training_results, plot_paths)

        # Ensure properties exist
        self._ensure_properties_exist(desired_properties)

        run_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        git_commit = self._get_git_commit()

        # Step 2: Upload plots
        drive_links = {}
        for plot_name, file_path in plot_paths.items():
            if os.path.exists(file_path):
                file_id = upload_file_to_drive(file_path, self.drive_folder_id)
                if file_id:
                    made_public = make_file_public(file_id)
                    if made_public:
                        public_url = get_public_link(file_id)  # Should return direct image URL
                        drive_links[plot_name] = public_url
                        logger.debug(f"Direct image URL for {plot_name}: {public_url}")
            else:
                logger.warning(f"Plot file not found: {file_path}")

        # Step 3: Build properties with separate columns
        page_properties = self._build_page_properties_separate_columns(
            run_date=run_date,
            git_commit=git_commit,
            config_dict=config_dict,
            training_results=training_results,
            model_path=model_path,
            drive_links=drive_links
        )

        # Step 4: Create page in Notion
        try:
            new_page = self.notion_client.pages.create(
                parent={"database_id": self.notion_db_id},
                properties=page_properties
            )
            logger.info(f"Created new Notion page: {new_page['id']}")
            return new_page
        except APIResponseError as e:
            logger.error(f"Failed to create new page: {e}")
            raise

    def _gather_desired_properties(self, config_dict: Dict[str, Any],
                                   training_results: Dict[str, Any],
                                   plot_paths: Dict[str, Any]) -> Dict[str, str]:
        """
        Gathers all desired properties with their types based on config and training results.
        :return: Dict where key is property name and value is property type.
        """
        desired = {}

        # Base properties
        desired["Date"] = "date"
        desired["Git Commit"] = "multi_select"
        desired["Model Path"] = "multi_select"

        # Config properties
        flat_config = self._flatten_config(config_dict)
        for key, val in flat_config.items():
            if isinstance(val, list):
                prop_name_size = f"{key} size"
                desired[prop_name_size] = "number"
                prop_name = f"{key}"
                desired[prop_name] = "multi_select"
            else:
                prop_name = f"{key}"
                desired[prop_name] = "rich_text"

        # Training results
        for metric, value in training_results.items():
            prop_name = f"{metric}"
            desired[prop_name] = "number" if isinstance(value, (int, float)) else "rich_text"

        # Plot links
        for plot_name in plot_paths.keys():
            prop_name = f"Plot: {plot_name}"
            desired[prop_name] = "files"  # Changed from 'url' to 'files'

        return desired

    def _flatten_config(self, config: Dict[str, Any], parent_key: str = '', sep: str = ' ') -> Dict[str, Any]:
        """
        Flattens a nested configuration dictionary.
        :param config: Nested config dictionary.
        :param parent_key: Base key string.
        :param sep: Separator between keys.
        :return: Flattened dictionary.
        """
        items = {}
        for k, v in config.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.update(self._flatten_config(v, new_key, sep=sep))
            else:
                items[new_key] = v
        return items

    def _build_page_properties_separate_columns(
        self,
        run_date: str,
        git_commit: str,
        config_dict: Dict[str, Any],
        training_results: Dict[str, Any],
        model_path: str,
        drive_links: Dict[str, str]
    ) -> Dict[str, Any]:
        """
        Creates a properties dict where each config key -> property name,
        each training_results key -> metric.
        Also includes run date, git commit, model path, etc.
        """
        props = {
            "Date": {
                "date": {
                    "start": run_date
                }
            },
            "Git Commit": {
                "multi_select": [
                    {"name": git_commit}
                ]
            },
            "Model Path": {
                "multi_select": [
                    {"name": model_path}
                ]
            }
        }

        # Flatten the config_dict
        flat_config = self._flatten_config(config_dict)

        # 1) For each config key
        for key, val in flat_config.items():
            if isinstance(val, list):
                # Define size property
                prop_name_size = f"{key} size"
                props[prop_name_size] = {
                    "number": len(val)
                }
                # Define multi-select property
                props[key] = {
                    "multi_select": [{"name": str(item)} for item in val]
                }
            else:
                # Define rich_text property
                prop_name = f"{key}"
                props[prop_name] = {
                    "rich_text": [
                        {"text": {"content": str(val)}}
                    ]
                }

        # 2) For each training result
        for metric, value in training_results.items():
            prop_name = f"{metric}"
            if isinstance(value, (int, float)):
                props[prop_name] = { "number": value }
            else:
                props[prop_name] = {
                    "rich_text": [
                        {"text": {"content": str(value)}}
                    ]
                }

        # 3) Plot links
        for plot_name, url in drive_links.items():
            # e.g. "Plot: true_vs_pred"
            prop_name = f"Plot: {plot_name}"
            props[prop_name] = {
                "files": [{
                    "type": "external",
                    "name": plot_name,
                    "external": {
                        "url": url
                    }
                }]
            }

        return props

    def _get_git_commit(self) -> str:
        try:
            commit_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
            return commit_hash
        except Exception as e:
            logger.warning(f"Could not get git commit: {e}")
            return "N/A"

    def _load_config(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, 'r') as f:
            return yaml.safe_load(f)



