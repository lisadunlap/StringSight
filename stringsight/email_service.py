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
    # NOTE: Brevo API support is temporarily disabled to ensure emails come from
    # the correct sender address (stringsightai@gmail.com instead of brevosend.com).
    # Brevo may be re-enabled in the future for better deliverability and higher sending limits.
    # To re-enable Brevo, uncomment the code block below and set BREVO_API_KEY in .env

    # Check for Brevo API Key (currently disabled - see note above)
    brevo_api_key = None  # os.getenv('BREVO_API_KEY')

    if brevo_api_key:
        logger.info("Using Brevo API for email sending")
        try:
            import requests
            import base64
            
            zip_path = create_results_zip(results_dir)
            
            # Read and encode the zip file
            with open(zip_path, "rb") as f:
                encoded_content = base64.b64encode(f.read()).decode()
            
            url = "https://api.brevo.com/v3/smtp/email"
            
            headers = {
                "accept": "application/json",
                "api-key": brevo_api_key,
                "content-type": "application/json"
            }
            
            payload = {
                "sender": {"email": sender_email},
                "to": [{"email": recipient_email}],
                "subject": "Your StringSight Results are Here!",
                "htmlContent": """
<html>
<body>
<p>Oh hello there,</p>

<p>Your StringSight clustering results are attached, get excited! 🎉</p>

<p>To view results, simply upload the zip file to <a href="https://stringsight.com">stringsight.com</a> (click the 'Load Results' button on the top right of the homepage)</p>

<p>The attached zip file contains all clustering outputs including:</p>
<ul>
<li>Original conversation data (conversations.jsonl)</li>
<li>Cluster definitions (clusters.jsonl)</li>
<li>Data properties (properties.jsonl)</li>
<li>Cluster scores and metrics (scores_df.jsonl files)</li>
</ul>

<p>Thank you for using StringSight! Hopefully you get some good insights from your strings. If you find this tool useful, please toss us a github star <a href="https://github.com/lisadunlap/StringSight">⭐ github.com/lisadunlap/StringSight</a></p>

<p>Best regards,<br>
Some Berkeley Folks</p>
</body>
</html>
""",
                "attachment": [
                    {
                        "content": encoded_content,
                        "name": f"{Path(zip_path).name}"
                    }
                ]
            }
            
            response = requests.post(url, headers=headers, json=payload)
            
            os.remove(zip_path)
            
            if response.status_code in [200, 201, 202]:
                logger.info(f"Results emailed successfully via Brevo to {recipient_email}")
                return {
                    'success': True,
                    'message': f'Results successfully sent to {recipient_email} via Brevo'
                }
            else:
                error_msg = f"Brevo API Error: {response.status_code} - {response.text}"
                logger.error(error_msg)
                return {
                    'success': False,
                    'message': error_msg
                }
                
        except Exception as e:
            logger.error(f"Failed to send email via Brevo: {str(e)}", exc_info=True)
            return {
                'success': False,
                'message': f'Failed to send email via Brevo: {str(e)}'
            }

    # Fallback to SMTP if no Brevo key
    smtp_server = smtp_server or settings.EMAIL_SMTP_SERVER
    smtp_port = smtp_port or settings.EMAIL_SMTP_PORT
    sender_email = sender_email or settings.EMAIL_SENDER
    sender_password = sender_password or settings.EMAIL_PASSWORD

    # Check for missing configuration
    missing_vars = []
    if not smtp_server: missing_vars.append("EMAIL_SMTP_SERVER")
    if not sender_email: missing_vars.append("EMAIL_SENDER")
    if not sender_password: missing_vars.append("EMAIL_PASSWORD")

    if missing_vars:
        error_msg = f"Email configuration missing: {', '.join(missing_vars)}. Please set these environment variables OR set BREVO_API_KEY."
        logger.error(error_msg)
        return {
            'success': False,
            'message': error_msg
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
        msg['Subject'] = f'Your StringSight Results are Here!'

        body = f"""
<html>
<body>
<p>Oh hello there,</p>

<p>Your StringSight clustering results are attached, get excited! 🎉</p>

<p>To view results, simply upload the zip file to <a href="https://stringsight.com">stringsight.com</a> (click the 'Load Results' button on the top right of the homepage)</p>

<p>The attached zip file contains all clustering outputs including:</p>
<ul>
<li>Original conversation data (conversations.jsonl)</li>
<li>Cluster definitions (clusters.jsonl)</li>
<li>Data properties (properties.jsonl)</li>
<li>Cluster scores and metrics (scores_df.jsonl files)</li>
</ul>

<p>Thank you for using StringSight! Hopefully you get some good insights from your strings. If you find this tool useful, please toss us a github star <a href="https://github.com/lisadunlap/StringSight">⭐ github.com/lisadunlap/StringSight</a></p>

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

        # Helper to resolve to IPv4 to avoid Docker IPv6 timeouts
        def get_ipv4_addr(host, port):
            try:
                import socket
                infos = socket.getaddrinfo(host, port, socket.AF_INET)
                if infos:
                    return infos[0][4][0]
            except Exception:
                pass
            return host

        server_ip = get_ipv4_addr(smtp_server, smtp_port)
        logger.info(f"Resolved {smtp_server} to {server_ip}")

        # Handle SSL vs STARTTLS based on port
        if smtp_port == 465:
            logger.info(f"Connecting to SMTP server {server_ip}:{smtp_port} using SSL")
            with smtplib.SMTP_SSL(server_ip, smtp_port) as server:
                server.login(sender_email, sender_password)
                server.send_message(msg)
        else:
            logger.info(f"Connecting to SMTP server {server_ip}:{smtp_port} using STARTTLS")
            with smtplib.SMTP(server_ip, smtp_port) as server:
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
        logger.error(f"Failed to send email: {str(e)}", exc_info=True)
        return {
            'success': False,
            'message': f'Failed to send email: {str(e)}'
        }
