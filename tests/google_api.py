from __future__ import print_function
import matplotlib.pyplot as plt
import os.path
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload

# Generate a simple plot
x = [1, 2, 3, 4]
y = [10, 20, 25, 30]
plt.plot(x, y)
plt.title('My Plot')
plt.xlabel('X Axis')
plt.ylabel('Y Axis')

# Save the plot locally
filename = 'my_plot.png'
plt.savefig(filename)
plt.close()

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/drive.file']

def get_drive_service():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.json'):
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                r'C:\Users\proxi\Documents\coding\goauth\client_secret_1058501869990-5nnvgvmsqnt61vkns6vd21v15798f8pn.apps.googleusercontent.com.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('drive', 'v3', credentials=creds)
    return service

drive_service = get_drive_service()


file_metadata = {
    'name': 'my_plot.png',  # The name to appear in Drive
    'parents': ['1s6n7YgoUoo8DHeYJqyaSR-l_TQlNeTLd']  # Optional: specify a Drive folder by ID
}
media = MediaFileUpload('my_plot.png', mimetype='image/png')
file = drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute()
print("File uploaded to Drive with ID:", file.get('id'))
