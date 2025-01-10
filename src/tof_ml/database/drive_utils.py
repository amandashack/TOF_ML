import os
import logging
import time
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2.service_account import Credentials

logger = logging.getLogger(__name__)

# Define the scope and path to your service account key file
SCOPES = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE = r'/sdf/home/a/ajshack/TOF_ML/tof-ml-001.json'  # Replace with your service account key file path

def get_drive_service():
    """
    Uses service account credentials to get a Google Drive service object.
    """
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)
    return service

def upload_file_to_drive(file_path: str, folder_id: str = None) -> str:
    service = get_drive_service()

    # Generate a unique suffix
    timestamp = int(time.time())  # or use str(uuid.uuid4()) for more uniqueness
    base_name = os.path.basename(file_path)
    unique_name = f"{os.path.splitext(base_name)[0]}_{timestamp}.png"  # Adjust extension as needed

    file_metadata = {
        'name': unique_name
    }
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, resumable=True)

    try:
        uploaded_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id'
        ).execute()
        file_id = uploaded_file.get('id')
        if not file_id:
            logger.error("Failed to upload file to Drive.")
            return None

        logger.info(f"Uploaded file to Drive with ID: {file_id}")
        return file_id
    except Exception as e:
        logger.error(f"Failed to upload file to Drive: {e}")
        return None

def make_file_public(file_id: str) -> bool:
    """
    Makes the file with file_id publicly readable.
    Returns True if successful.
    """
    service = get_drive_service()
    try:
        permission = {
            'type': 'anyone',
            'role': 'reader'
        }
        service.permissions().create(
            fileId=file_id,
            body=permission
        ).execute()
        logger.info(f"File {file_id} is now publicly accessible.")
        return True
    except Exception as e:
        logger.error(f"Error making file {file_id} public: {e}")
        return False

def get_public_link(file_id: str) -> str:
    """
    Returns a publicly viewable link for the file.
    """
    # Direct image link format for Google Drive
    return f"https://drive.google.com/uc?export=view&id={file_id}"

