#!/bin/bash
# Deploy optimized nginx configuration for StringSight API
# Run this script ON YOUR DIGITAL OCEAN DROPLET (not locally!)

set -e  # Exit on error

echo "🚀 Deploying optimized nginx configuration for StringSight..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ Please run as root or with sudo${NC}"
    exit 1
fi

# 1. Find current nginx config
echo -e "${YELLOW}📁 Looking for existing nginx config...${NC}"
if [ -f /etc/nginx/sites-available/stringsight ]; then
    NGINX_CONFIG="/etc/nginx/sites-available/stringsight"
elif [ -f /etc/nginx/sites-available/api.stringsight.com ]; then
    NGINX_CONFIG="/etc/nginx/sites-available/api.stringsight.com"
elif [ -f /etc/nginx/conf.d/stringsight.conf ]; then
    NGINX_CONFIG="/etc/nginx/conf.d/stringsight.conf"
else
    # Create new config
    NGINX_CONFIG="/etc/nginx/sites-available/stringsight"
    echo -e "${YELLOW}⚠️  No existing config found. Creating new one at ${NGINX_CONFIG}${NC}"
fi

# 2. Backup existing config
if [ -f "$NGINX_CONFIG" ]; then
    BACKUP_FILE="${NGINX_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}💾 Backing up existing config to ${BACKUP_FILE}${NC}"
    cp "$NGINX_CONFIG" "$BACKUP_FILE"
fi

# 3. Get Vercel domain
echo -e "${YELLOW}🌐 What is your Vercel frontend domain?${NC}"
echo "   Examples: https://stringsight.vercel.app, https://app.stringsight.com"
read -p "   Enter domain: " VERCEL_DOMAIN

# Remove trailing slash if present
VERCEL_DOMAIN="${VERCEL_DOMAIN%/}"

# Validate it starts with https://
if [[ ! "$VERCEL_DOMAIN" =~ ^https:// ]]; then
    echo -e "${RED}❌ Domain must start with https://${NC}"
    exit 1
fi

# 4. Create optimized nginx config
echo -e "${YELLOW}✍️  Writing optimized nginx config...${NC}"
cat > "$NGINX_CONFIG" << 'NGINX_CONFIG_EOF'
# Nginx configuration for StringSight API (api.stringsight.com)
# This config optimizes for Vercel → Digital Ocean connections

# Upstream configuration with keepalive for better performance
upstream stringsight_api {
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;

    # Keep connections alive between nginx and uvicorn
    keepalive 32;
    keepalive_timeout 60s;
}

# HTTP → HTTPS redirect
server {
    listen 80;
    listen [::]:80;
    server_name api.stringsight.com;

    # Allow certbot challenges
    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    # Redirect all HTTP to HTTPS
    return 301 https://$host$request_uri;
}

# HTTPS server
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name api.stringsight.com;

    # SSL certificates (managed by certbot)
    ssl_certificate /etc/letsencrypt/live/api.stringsight.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.stringsight.com/privkey.pem;

    # SSL optimization
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers on;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384';

    # CRITICAL: CORS headers for Vercel frontend
    add_header 'Access-Control-Allow-Origin' 'VERCEL_DOMAIN_PLACEHOLDER' always;
    add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
    add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
    add_header 'Access-Control-Expose-Headers' 'Content-Length,Content-Range' always;
    add_header 'Access-Control-Allow-Credentials' 'true' always;

    # Handle CORS preflight requests (this is KEY for performance!)
    if ($request_method = 'OPTIONS') {
        add_header 'Access-Control-Allow-Origin' 'VERCEL_DOMAIN_PLACEHOLDER' always;
        add_header 'Access-Control-Allow-Methods' 'GET, POST, PUT, DELETE, OPTIONS' always;
        add_header 'Access-Control-Allow-Headers' 'DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization' always;
        add_header 'Access-Control-Max-Age' 86400 always;  # Cache preflight for 24 hours
        add_header 'Content-Type' 'text/plain; charset=utf-8' always;
        add_header 'Content-Length' 0 always;
        return 204;
    }

    # Increase client body size for file uploads
    client_max_body_size 100M;

    # Timeouts optimized for long-running clustering jobs
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;

    # Buffer settings to reduce memory usage
    proxy_buffering on;
    proxy_buffer_size 4k;
    proxy_buffers 8 4k;
    proxy_busy_buffers_size 8k;

    # Health check endpoint - respond quickly
    location = /health {
        proxy_pass http://stringsight_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        # Aggressive timeout for health checks
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;

        # CORS headers still needed
        add_header 'Access-Control-Allow-Origin' 'VERCEL_DOMAIN_PLACEHOLDER' always;
    }

    location = /api/health {
        proxy_pass http://stringsight_api;
        proxy_http_version 1.1;
        proxy_set_header Connection "";

        # Aggressive timeout for health checks
        proxy_connect_timeout 5s;
        proxy_send_timeout 5s;
        proxy_read_timeout 5s;

        # CORS headers still needed
        add_header 'Access-Control-Allow-Origin' 'VERCEL_DOMAIN_PLACEHOLDER' always;
    }

    # All other API endpoints
    location / {
        proxy_pass http://stringsight_api;
        proxy_http_version 1.1;

        # Connection pooling headers
        proxy_set_header Connection "";

        # Standard proxy headers
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Enable TCP keepalive at proxy level
        proxy_socket_keepalive on;
    }

    # Logging
    access_log /var/log/nginx/stringsight-access.log;
    error_log /var/log/nginx/stringsight-error.log warn;
}
NGINX_CONFIG_EOF

# Replace placeholder with actual Vercel domain
sed -i "s|VERCEL_DOMAIN_PLACEHOLDER|$VERCEL_DOMAIN|g" "$NGINX_CONFIG"

# 5. Enable site if using sites-available
if [[ "$NGINX_CONFIG" == /etc/nginx/sites-available/* ]]; then
    SITE_NAME=$(basename "$NGINX_CONFIG")
    echo -e "${YELLOW}🔗 Enabling site ${SITE_NAME}...${NC}"
    ln -sf "$NGINX_CONFIG" "/etc/nginx/sites-enabled/$SITE_NAME"
fi

# 6. Test nginx configuration
echo -e "${YELLOW}🧪 Testing nginx configuration...${NC}"
if nginx -t; then
    echo -e "${GREEN}✅ Nginx configuration is valid${NC}"
else
    echo -e "${RED}❌ Nginx configuration test failed!${NC}"
    echo -e "${YELLOW}💾 Restoring backup...${NC}"
    if [ -f "$BACKUP_FILE" ]; then
        cp "$BACKUP_FILE" "$NGINX_CONFIG"
        echo -e "${GREEN}✅ Backup restored${NC}"
    fi
    exit 1
fi

# 7. Reload nginx
echo -e "${YELLOW}🔄 Reloading nginx...${NC}"
systemctl reload nginx

echo ""
echo -e "${GREEN}✅ Nginx configuration deployed successfully!${NC}"
echo ""
echo -e "${YELLOW}📊 Next steps:${NC}"
echo "   1. Test health check from your local machine:"
echo "      curl -I https://api.stringsight.com/health"
echo ""
echo "   2. Test CORS preflight:"
echo "      curl -I -X OPTIONS https://api.stringsight.com/health \\"
echo "        -H \"Origin: $VERCEL_DOMAIN\" \\"
echo "        -H \"Access-Control-Request-Method: GET\""
echo ""
echo "   3. Deploy your frontend changes to Vercel"
echo ""
echo "   4. Monitor nginx logs for errors:"
echo "      tail -f /var/log/nginx/stringsight-error.log"
echo ""
echo -e "${GREEN}🎉 Your backend should now connect faster from Vercel!${NC}"
