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

logger = logging.getLogger(__name__)


def create_results_zip(results_dir: str) -> str:
    """
    Create a zip file of the results directory.

    Args:
        results_dir: Path to the results directory to zip

    Returns:
        Path to the created zip file
    """
    temp_dir = tempfile.gettempdir()
    zip_path = os.path.join(temp_dir, f"{Path(results_dir).name}.zip")

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(results_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(results_dir))
                zipf.write(file_path, arcname)

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
    smtp_server = smtp_server or os.getenv('EMAIL_SMTP_SERVER')
    smtp_port = smtp_port or int(os.getenv('EMAIL_SMTP_PORT', 587))
    sender_email = sender_email or os.getenv('EMAIL_SENDER')
    sender_password = sender_password or os.getenv('EMAIL_PASSWORD')

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
        msg['Subject'] = f'StringSight Clustering Results - {experiment_name}'

        body = f"""
Hello,

Your StringSight clustering results for experiment "{experiment_name}" are attached.

The attached zip file contains all clustering outputs including:
- Cluster definitions (clusters.jsonl)
- Data properties (properties.jsonl)
- Cluster scores and metrics
- Embeddings

Thank you for using StringSight!

Best regards,
StringSight Team
"""

        msg.attach(MIMEText(body, 'plain'))

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
