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
    
    # Try connecting
    try:
        connect_to_server(smtp_server, port, sender_email, sender_password, recipient)
        print("\n✨ Configuration Verified Successfully!")
        return True
    except Exception as e:
        print(f"\n❌ Connection Failed: {e}")
        
        # If failed, try forcing IPv4 if it looks like a network unreachable error
        if "unreachable" in str(e).lower() or "101" in str(e):
            print("\n⚠️  Network Unreachable. Attempting to force IPv4...")
            try:
                # Resolve to IPv4 address manually
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
