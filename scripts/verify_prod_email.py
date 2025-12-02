#!/usr/bin/env python3
"""
Verify Production Email Configuration
Run this script inside your Docker container to debug email issues.

Usage:
    python scripts/verify_prod_email.py [recipient_email]

Example:
    python scripts/verify_prod_email.py me@example.com
"""
import os
import sys
import smtplib
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def mask_secret(secret):
    if not secret:
        return "Not Set"
    if len(secret) < 4:
        return "***"
    return f"{secret[:2]}***{secret[-2:]}"

def verify_email_config(recipient=None):
    # Load env vars
    load_dotenv()
    
    smtp_server = os.getenv('EMAIL_SMTP_SERVER')
    smtp_port = os.getenv('EMAIL_SMTP_PORT')
    sender_email = os.getenv('EMAIL_SENDER')
    sender_password = os.getenv('EMAIL_PASSWORD')
    
    print("\n" + "="*50)
    print("📧 Email Configuration Check")
    print("="*50)
    
    print(f"SMTP Server:   {smtp_server}")
    print(f"SMTP Port:     {smtp_port}")
    print(f"Sender Email:  {sender_email}")
    print(f"Password:      {mask_secret(sender_password)}")
    
    # Check for missing vars
    missing = []
    if not smtp_server: missing.append('EMAIL_SMTP_SERVER')
    if not smtp_port: missing.append('EMAIL_SMTP_PORT')
    if not sender_email: missing.append('EMAIL_SENDER')
    if not sender_password: missing.append('EMAIL_PASSWORD')
    
    if missing:
        print(f"\n❌ Missing required environment variables: {', '.join(missing)}")
        return False
        
    try:
        port = int(smtp_port)
    except ValueError:
        print(f"\n❌ Invalid port number: {smtp_port}")
        return False

    print("\n🔄 Testing Connection...")
    
    try:
        if port == 465:
            print(f"   Connecting to {smtp_server}:{port} using SSL...")
            with smtplib.SMTP_SSL(smtp_server, port) as server:
                print("   ✅ Connected (SSL)")
                print("   Logging in...")
                server.login(sender_email, sender_password)
                print("   ✅ Login Successful")
                
                if recipient:
                    print(f"   Sending test email to {recipient}...")
                    msg = f"Subject: StringSight Test Email\n\nThis is a test email from the verification script."
                    server.sendmail(sender_email, recipient, msg)
                    print("   ✅ Email Sent")
        else:
            print(f"   Connecting to {smtp_server}:{port} using STARTTLS...")
            with smtplib.SMTP(smtp_server, port) as server:
                print("   ✅ Connected")
                print("   Starting TLS...")
                server.starttls()
                print("   ✅ TLS Started")
                print("   Logging in...")
                server.login(sender_email, sender_password)
                print("   ✅ Login Successful")
                
                if recipient:
                    print(f"   Sending test email to {recipient}...")
                    msg = f"Subject: StringSight Test Email\n\nThis is a test email from the verification script."
                    server.sendmail(sender_email, recipient, msg)
                    print("   ✅ Email Sent")
                    
        print("\n✨ Configuration Verified Successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    recipient = sys.argv[1] if len(sys.argv) > 1 else None
    verify_email_config(recipient)
