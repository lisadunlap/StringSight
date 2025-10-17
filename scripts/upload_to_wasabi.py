#!/usr/bin/env python3
"""
Script to upload folders to Wasabi S3-compatible storage.

Usage:
    python upload_to_wasabi.py --source /path/to/results --bucket your-bucket-name
    python upload_to_wasabi.py --source results --bucket your-bucket-name --prefix my-experiment
    python upload_to_wasabi.py --source results --bucket your-bucket-name --dry-run
"""

import argparse
import os
import sys
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from pathlib import Path
import logging
from tqdm import tqdm
import mimetypes

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WasabiUploader:
    def __init__(self, access_key=None, secret_key=None, region='us-east-1', endpoint_url=None):
        """
        Initialize Wasabi uploader.
        
        Args:
            access_key: Wasabi access key (if None, will use environment variables)
            secret_key: Wasabi secret key (if None, will use environment variables)
            region: Wasabi region (default: us-east-1)
            endpoint_url: Custom endpoint URL (if None, uses default Wasabi endpoint)
        """
        self.region = region
        
        # Set default Wasabi endpoint if not provided
        if endpoint_url is None:
            self.endpoint_url = f"https://s3.{region}.wasabisys.com"
        else:
            self.endpoint_url = endpoint_url
            
        # Initialize S3 client
        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name=region,
                endpoint_url=self.endpoint_url
            )
            logger.info(f"Initialized Wasabi client for region: {region}")
        except NoCredentialsError:
            logger.error("No credentials found. Please set WASABI_ACCESS_KEY and WASABI_SECRET_KEY environment variables.")
            sys.exit(1)
    
    def upload_file(self, local_path, bucket, s3_key, dry_run=False):
        """
        Upload a single file to Wasabi.
        
        Args:
            local_path: Local file path
            bucket: Wasabi bucket name
            s3_key: S3 key (path in bucket)
            dry_run: If True, don't actually upload
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if dry_run:
                logger.info(f"[DRY RUN] Would upload: {local_path} -> s3://{bucket}/{s3_key}")
                return True
                
            # Determine content type
            content_type, _ = mimetypes.guess_type(local_path)
            if content_type is None:
                content_type = 'application/octet-stream'
            
            # Upload file
            self.s3_client.upload_file(
                local_path,
                bucket,
                s3_key,
                ExtraArgs={'ContentType': content_type}
            )
            logger.info(f"Uploaded: {local_path} -> s3://{bucket}/{s3_key}")
            return True
            
        except ClientError as e:
            logger.error(f"Failed to upload {local_path}: {e}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error uploading {local_path}: {e}")
            return False
    
    def upload_folder(self, local_folder, bucket, prefix="", dry_run=False, exclude_patterns=None):
        """
        Upload an entire folder to Wasabi.
        
        Args:
            local_folder: Local folder path
            bucket: Wasabi bucket name
            prefix: S3 prefix (folder path in bucket)
            dry_run: If True, don't actually upload
            exclude_patterns: List of patterns to exclude (e.g., ['*.tmp', '__pycache__'])
            
        Returns:
            tuple: (successful_uploads, failed_uploads, total_files)
        """
        local_path = Path(local_folder)
        if not local_path.exists():
            logger.error(f"Local folder does not exist: {local_folder}")
            return 0, 0, 0
        
        if not local_path.is_dir():
            logger.error(f"Path is not a directory: {local_folder}")
            return 0, 0, 0
        
        # Check if bucket exists
        try:
            self.s3_client.head_bucket(Bucket=bucket)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"Bucket '{bucket}' does not exist")
                return 0, 0, 0
            elif error_code == '403':
                logger.error(f"Access denied to bucket '{bucket}'")
                return 0, 0, 0
            else:
                logger.error(f"Error accessing bucket '{bucket}': {e}")
                return 0, 0, 0
        
        successful_uploads = 0
        failed_uploads = 0
        total_files = 0
        
        # Collect all files to upload
        files_to_upload = []
        for root, dirs, files in os.walk(local_path):
            # Skip excluded directories
            if exclude_patterns:
                dirs[:] = [d for d in dirs if not any(pattern in d for pattern in exclude_patterns)]
            
            for file in files:
                file_path = Path(root) / file
                
                # Skip excluded files
                if exclude_patterns:
                    if any(pattern in str(file_path) for pattern in exclude_patterns):
                        continue
                
                # Calculate S3 key
                relative_path = file_path.relative_to(local_path)
                s3_key = str(relative_path)
                if prefix:
                    s3_key = f"{prefix}/{s3_key}"
                
                files_to_upload.append((file_path, s3_key))
                total_files += 1
        
        logger.info(f"Found {total_files} files to upload")
        
        if dry_run:
            logger.info("DRY RUN MODE - No files will be actually uploaded")
            for file_path, s3_key in files_to_upload:
                logger.info(f"[DRY RUN] Would upload: {file_path} -> s3://{bucket}/{s3_key}")
            return total_files, 0, total_files
        
        # Upload files with progress bar
        with tqdm(total=total_files, desc="Uploading files") as pbar:
            for file_path, s3_key in files_to_upload:
                if self.upload_file(str(file_path), bucket, s3_key, dry_run=False):
                    successful_uploads += 1
                else:
                    failed_uploads += 1
                pbar.update(1)
        
        return successful_uploads, failed_uploads, total_files

def main():
    parser = argparse.ArgumentParser(description='Upload folders to Wasabi S3-compatible storage')
    parser.add_argument('--source', required=True, help='Source folder path to upload')
    parser.add_argument('--bucket', default='vibes', help='Wasabi bucket name')
    parser.add_argument('--prefix', default='', help='S3 prefix (folder path in bucket)')
    parser.add_argument('--access-key', help='Wasabi access key (or set WASABI_ACCESS_KEY env var)')
    parser.add_argument('--secret-key', help='Wasabi secret key (or set WASABI_SECRET_ACCESS_KEY env var)')
    parser.add_argument('--region', default='us-west-1', help='Wasabi region (default: us-east-1)')
    parser.add_argument('--endpoint-url', help='Custom endpoint URL')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be uploaded without actually uploading')
    parser.add_argument('--exclude', nargs='*', default=['__pycache__', '*.tmp', '.DS_Store'], 
                       help='Patterns to exclude (default: __pycache__, *.tmp, .DS_Store)')
    
    args = parser.parse_args()
    
    # Get credentials from environment if not provided
    access_key = args.access_key or os.getenv('WASABI_ACCESS_KEY')
    secret_key = args.secret_key or os.getenv('WASABI_SECRET_ACCESS_KEY')
    
    if not access_key or not secret_key:
        logger.error("Please provide Wasabi credentials via --access-key/--secret-key arguments or WASABI_ACCESS_KEY/WASABI_SECRET_KEY environment variables")
        sys.exit(1)
    
    # Initialize uploader
    uploader = WasabiUploader(
        access_key=access_key,
        secret_key=secret_key,
        region=args.region,
        endpoint_url=args.endpoint_url
    )
    
    # Upload folder
    logger.info(f"Starting upload of '{args.source}' to bucket '{args.bucket}'")
    if args.prefix:
        logger.info(f"Using prefix: {args.prefix}")
    
    successful, failed, total = uploader.upload_folder(
        local_folder=args.source,
        bucket=args.bucket,
        prefix=args.prefix,
        dry_run=args.dry_run,
        exclude_patterns=args.exclude
    )
    
    # Summary
    logger.info(f"Upload complete!")
    logger.info(f"Total files: {total}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    main() 