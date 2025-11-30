import os
import smtplib
import zipfile
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from pathlib import Path
import tempfile
import logging

from stringsight.config import settings

logger = logging.getLogger(__name__)


def create_results_zip(results_dir: str, max_size_mb: int = 24) -> str:
    """
    Create a zip file of the results directory, excluding large redundant files.

    Args:
        results_dir: Path to the results directory to zip
        max_size_mb: Maximum size in MB before warning (default 24MB for Gmail)

    Returns:
        Path to the created zip file
    """
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, f"{Path(results_dir).name}.zip")
    
    # Files to exclude to save space (redundant with jsonl files)
    exclude_files = {'full_dataset.json', 'full_dataset.parquet'}
    exclude_extensions = {'.parquet', '.pkl', '.pickle'}

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                # Skip excluded files
                if file in exclude_files or os.path.splitext(file)[1] in exclude_extensions:
                    continue
                    
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(results_dir))
                zipf.write(file_path, arcname)
                
    # Check size
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    if size_mb > max_size_mb:
        logger.warning(f"⚠️ Created zip file is {size_mb:.2f}MB, which may exceed email limits ({max_size_mb}MB)")
        
    return zip_path


def send_results_email(
    recipient_email: str,
    results_dir: str,
    experiment_name: str,
    smtp_server: str = None,
    smtp_port: int = None,
    sender_email: str = None,
    sender_password: str = None
) -> dict:
    """
    Send clustering results to a recipient via email.

    Args:
        recipient_email: Email address to send results to
        results_dir: Path to the results directory
        experiment_name: Name of the experiment/clustering run
        smtp_server: SMTP server address (defaults to env var EMAIL_SMTP_SERVER)
        smtp_port: SMTP port (defaults to env var EMAIL_SMTP_PORT or 587)
        sender_email: Sender email address (defaults to env var EMAIL_SENDER)
        sender_password: Sender email password (defaults to env var EMAIL_PASSWORD)

    Returns:
        Dict with 'success' boolean and 'message' string
    """
    smtp_server = smtp_server or settings.EMAIL_SMTP_SERVER
    smtp_port = smtp_port or settings.EMAIL_SMTP_PORT
    sender_email = sender_email or settings.EMAIL_SENDER
    sender_password = sender_password or settings.EMAIL_PASSWORD

    if not all([smtp_server, sender_email, sender_password]):
        return {
            'success': False,
            'message': 'Email configuration missing. Please set EMAIL_SMTP_SERVER, EMAIL_SENDER, and EMAIL_PASSWORD environment variables.'
        }

    if not os.path.exists(results_dir):
        return {
            'success': False,
            'message': f'Results directory not found: {results_dir}'
        }

    try:
        zip_path = create_results_zip(results_dir)

        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = f'Your StringSight Clustering Results are Here!'

        body = f"""
<html>
<body>
<p>Oh hello there,</p>

<p>Your StringSight clustering results for experiment "{experiment_name}" are attached, get excited! 🎉</p>

<p>To view results, simply upload the zip file to <a href="https://stringsight.com">stringsight.com</a> (click the 'Load Results' button on the top right of the homepage)</p>

<p>The attached zip file contains all clustering outputs including:</p>
<ul>
<li>Original conversation data (conversations.jsonl)</li>
<li>Cluster definitions (clusters.jsonl)</li>
<li>Data properties (properties.jsonl)</li>
<li>Cluster scores and metrics (scores_df.jsonl files)</li>
</ul>

<p>Thank you for using StringSight! If you find this tool useful, we take funding in the form of github stars <a href="https://github.com/lisadunlap/StringSight">⭐ github.com/lisadunlap/StringSight</a></p>

<p>Best regards,<br>
Some Berkeley Folks</p>
</body>
</html>
"""

        msg.attach(MIMEText(body, 'html'))

        with open(zip_path, 'rb') as attachment:
            part = MIMEBase('application', 'zip')
            part.set_payload(attachment.read())
            encoders.encode_base64(part)
            part.add_header(
                'Content-Disposition',
                f'attachment; filename={Path(zip_path).name}'
            )
            msg.attach(part)

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)

        os.remove(zip_path)

        logger.info(f"Results emailed successfully to {recipient_email}")
        return {
            'success': True,
            'message': f'Results successfully sent to {recipient_email}'
        }

    except Exception as e:
        logger.error(f"Failed to send email: {str(e)}")
        return {
            'success': False,
            'message': f'Failed to send email: {str(e)}'
        }
