# src/utils/google_drive_uploader.py

import os
import logging
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

logger = logging.getLogger(__name__)

SCOPES = ['https://www.googleapis.com/auth/drive.file']
TOKEN_FILE = 'token.json'  # Where we store user credentials
CLIENT_SECRET_FILE = r'C:\Users\proxi\Documents\coding\goauth\client_secret_1058501869990-5nnvgvmsqnt61vkns6vd21v15798f8pn.apps.googleusercontent.com.json'

def get_drive_service():
    """
    Uses OAuth user flow to get a Google Drive service object.
    If token.json exists and is valid, reuses it; otherwise, prompts the user to log in.
    """
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired token...")
            creds.refresh(Request())
        else:
            logger.info("Fetching new tokens via OAuth flow...")
            flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    return service

def upload_file_to_drive(file_path: str, folder_id: str = None) -> str:
    """
    Uploads the file at file_path to Google Drive.
    Optionally provide a folder_id to place the file in a specific folder.
    Returns the newly created file ID.
    """
    service = get_drive_service()
    file_metadata = {
        'name': os.path.basename(file_path)
    }
    if folder_id:
        file_metadata['parents'] = [folder_id]

    media = MediaFileUpload(file_path, resumable=True)
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
    # One known shareable link pattern:
    # https://drive.google.com/uc?id={file_id}&export=view
    return f"https://drive.google.com/uc?id={file_id}&export=view"