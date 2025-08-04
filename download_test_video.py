#!/usr/bin/env python3
"""
Download test video from Google Drive for vast.ai testing
"""
import requests
import os
from loguru import logger

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive given its file ID."""
    URL = "https://docs.google.com/uc?export=download"
    
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)
    
    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
    
    save_response_content(response, destination)

def get_confirm_token(response):
    """Extract confirmation token from Google Drive response."""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save the response content to a file."""
    CHUNK_SIZE = 32768
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

def main():
    # Extract file ID from your Google Drive link
    # https://drive.google.com/file/d/1QAOVhgw7_0YT4L2Bfu4LPibEfgahLBj8/view?usp=share_link
    file_id = "1QAOVhgw7_0YT4L2Bfu4LPibEfgahLBj8"
    
    # Create test_videos directory if it doesn't exist
    os.makedirs("test_videos", exist_ok=True)
    
    destination = "test_videos/test_video.mp4"
    
    logger.info(f"Downloading video from Google Drive...")
    logger.info(f"File ID: {file_id}")
    logger.info(f"Destination: {destination}")
    
    try:
        download_file_from_google_drive(file_id, destination)
        
        # Check if file was downloaded successfully
        if os.path.exists(destination) and os.path.getsize(destination) > 0:
            file_size = os.path.getsize(destination) / (1024 * 1024)  # Size in MB
            logger.success(f"✅ Video downloaded successfully!")
            logger.info(f"File size: {file_size:.2f} MB")
            logger.info(f"Saved to: {destination}")
        else:
            logger.error("❌ Download failed - file is empty or doesn't exist")
            
    except Exception as e:
        logger.error(f"❌ Download failed: {e}")

if __name__ == "__main__":
    main()