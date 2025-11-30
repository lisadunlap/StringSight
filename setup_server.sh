#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting Server Setup..."

# 1. Update System
echo "📦 Updating system packages..."
sudo apt-get update
sudo apt-get upgrade -y

# 2. Install Docker & Docker Compose
echo "🐳 Installing Docker..."
sudo apt-get install -y docker.io docker-compose docker-buildx

# 3. Start Docker and enable it to run on boot
echo "🔌 Enabling Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# 4. Add current user to docker group (so you don't need 'sudo' for docker commands)
# Note: You'll need to logout and login again for this to take effect
echo "👤 Adding user to docker group..."
sudo usermod -aG docker $USER

echo "✅ Setup Complete!"
echo "------------------------------------------------"
echo "Next steps:"
echo "1. Logout and log back in: 'exit' then ssh back in"
echo "2. Clone your repo: 'git clone <your-repo-url>'"
echo "3. Enter directory: 'cd StringSightNew'"
echo "4. Create .env file: 'nano .env' (paste your secrets)"
echo "5. Run app: 'docker-compose up -d --build'"
echo "------------------------------------------------"
