"""
Vercel entrypoint for FastAPI backend.
This file is required for Vercel to detect and deploy the FastAPI app.
"""
from stringsight.api import app

# Vercel looks for 'app' variable
__all__ = ['app']

