#!/usr/bin/env python3
"""
Verify Production Email Configuration & Network
Run this script inside your Docker container to debug email and network issues.

Usage:
    python scripts/verify_prod_email.py [recipient_email]

Example:
    python scripts/verify_prod_email.py me@example.com
"""
import os
import sys
import smtplib
import socket
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

def check_network():
    print("\n" + "="*50)
    print("🌐 Network Diagnostic")
    print("="*50)
    
    # 1. Check DNS Resolution
    print("1. DNS Resolution (smtp.gmail.com):")
    try:
        infos = socket.getaddrinfo("smtp.gmail.com", 587)
        for family, type, proto, canonname, sockaddr in infos:
            fam_str = "IPv4" if family == socket.AF_INET else "IPv6" if family == socket.AF_INET6 else str(family)
            print(f"   - {fam_str}: {sockaddr[0]}")
    except Exception as e:
        print(f"   ❌ DNS Lookup Failed: {e}")

    # 2. Check Outbound Connectivity (Google DNS)
    print("\n2. Outbound Connectivity (8.8.8.8:53):")
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=5)
        print("   ✅ Reachable")
    except Exception as e:
        print(f"   ❌ Unreachable: {e}")

def verify_email_config(recipient=None):
    # Load env vars
    load_dotenv()
    
    brevo_key = os.getenv('BREVO_API_KEY')
    smtp_server = os.getenv('EMAIL_SMTP_SERVER')
    smtp_port = os.getenv('EMAIL_SMTP_PORT')
    sender_email = os.getenv('EMAIL_SENDER')
    sender_password = os.getenv('EMAIL_PASSWORD')
    
    print("\n" + "="*50)
    print("📧 Email Configuration Check")
    print("="*50)
    
    if brevo_key:
        print("✅ Brevo API Key found!")
        print(f"Key:           {mask_secret(brevo_key)}")
        print(f"Sender Email:  {sender_email}")
        
        if not sender_email:
            print("❌ EMAIL_SENDER is missing (required for Brevo)")
            return False
            
        if recipient:
            print("\n🔄 Testing Brevo API...")
            try:
                import requests
                url = "https://api.brevo.com/v3/smtp/email"
                headers = {
                    "accept": "application/json",
                    "api-key": brevo_key,
                    "content-type": "application/json"
                }
                payload = {
                    "sender": {"email": sender_email},
                    "to": [{"email": recipient}],
                    "subject": "StringSight Brevo Test",
                    "htmlContent": "<p>This is a test email from StringSight using Brevo API.</p>"
                }
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code in [200, 201, 202]:
                    print("   ✅ Email sent successfully via Brevo!")
                    return True
                else:
                    print(f"   ❌ Brevo API Error: {response.status_code}")
                    print(f"   {response.text}")
                    return False
            except Exception as e:
                print(f"   ❌ Brevo Test Failed: {e}")
                return False
        else:
            print("\nℹ️  To test sending, provide a recipient email:")
            print("    python scripts/verify_prod_email.py me@example.com")
            return True

    # Fallback to SMTP checks if no Brevo key
    print("\nℹ️  No Brevo API Key found. Checking SMTP configuration...")
    
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
    
    # Try connecting
    try:
        connect_to_server(smtp_server, port, sender_email, sender_password, recipient)
        print("\n✨ Configuration Verified Successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")
        
        # If failed, try forcing IPv4 if it looks like a network unreachable error
        if "unreachable" in str(e).lower() or "101" in str(e) or "timed out" in str(e).lower():
            print("\n⚠️  Network Issue Detected. Attempting diagnostics...")
            
            # 1. Try IPv4 Force
            print("\n   [Attempt 1] Forcing IPv4...")
            try:
                ipv4_addr = None
                infos = socket.getaddrinfo(smtp_server, port, socket.AF_INET)
                if infos:
                    ipv4_addr = infos[0][4][0]
                    print(f"   Resolved {smtp_server} to IPv4: {ipv4_addr}")
                    connect_to_server(ipv4_addr, port, sender_email, sender_password, recipient)
                    print("\n✨ Success using IPv4! You may need to disable IPv6 or force IPv4 in your app.")
                    return True
                else:
                    print("   ❌ Could not resolve to IPv4 address.")
            except Exception as e2:
                print(f"   ❌ IPv4 Force Failed: {e2}")

            # 2. Try Port 465 (SSL)
            if port != 465:
                print("\n   [Attempt 2] Trying Port 465 (SSL)...")
                print("   (DigitalOcean often blocks port 587 but allows 465)")
                try:
                    # Resolve IPv4 for this too
                    ipv4_addr = None
                    infos = socket.getaddrinfo(smtp_server, 465, socket.AF_INET)
                    if infos:
                        ipv4_addr = infos[0][4][0]
                    else:
                        ipv4_addr = smtp_server
                        
                    connect_to_server(ipv4_addr, 465, sender_email, sender_password, recipient)
                    print("\n✨ Success using Port 465! Please update EMAIL_SMTP_PORT=465 in your .env")
                    return True
                except Exception as e3:
                    print(f"   ❌ Port 465 Failed: {e3}")

        import traceback
        traceback.print_exc()
        return False

def connect_to_server(server_addr, port, email, password, recipient):
    if port == 465:
        print(f"   Connecting to {server_addr}:{port} using SSL...")
        with smtplib.SMTP_SSL(server_addr, port) as server:
            print("   ✅ Connected (SSL)")
            server.login(email, password)
            print("   ✅ Login Successful")
            if recipient:
                send_test_msg(server, email, recipient)
    else:
        print(f"   Connecting to {server_addr}:{port} using STARTTLS...")
        with smtplib.SMTP(server_addr, port) as server:
            print("   ✅ Connected")
            server.starttls()
            print("   ✅ TLS Started")
            server.login(email, password)
            print("   ✅ Login Successful")
            if recipient:
                send_test_msg(server, email, recipient)

def send_test_msg(server, sender, recipient):
    print(f"   Sending test email to {recipient}...")
    msg = f"Subject: StringSight Test Email\n\nThis is a test email from the verification script."
    server.sendmail(sender, recipient, msg)
    print("   ✅ Email Sent")

if __name__ == "__main__":
    check_network()
    recipient = sys.argv[1] if len(sys.argv) > 1 else None
    verify_email_config(recipient)
