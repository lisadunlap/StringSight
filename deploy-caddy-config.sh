#!/bin/bash
# Deploy optimized Caddy configuration for StringSight API
# Run this script ON YOUR DIGITAL OCEAN DROPLET (not locally!)

set -e  # Exit on error

echo "🚀 Deploying optimized Caddy configuration for StringSight..."

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

# 1. Check if Caddy is installed
if ! command -v caddy &> /dev/null; then
    echo -e "${RED}❌ Caddy is not installed!${NC}"
    echo -e "${YELLOW}Install Caddy first: https://caddyserver.com/docs/install${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Caddy is installed ($(caddy version))${NC}"

# 2. Find Caddyfile location
CADDY_CONFIG="/etc/caddy/Caddyfile"
if [ ! -f "$CADDY_CONFIG" ]; then
    echo -e "${YELLOW}⚠️  Caddyfile not found at $CADDY_CONFIG${NC}"
    echo -e "${YELLOW}Creating new Caddyfile...${NC}"
    mkdir -p /etc/caddy
fi

# 3. Backup existing Caddyfile
if [ -f "$CADDY_CONFIG" ]; then
    BACKUP_FILE="${CADDY_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
    echo -e "${YELLOW}💾 Backing up existing Caddyfile to ${BACKUP_FILE}${NC}"
    cp "$CADDY_CONFIG" "$BACKUP_FILE"
fi

# 4. Get Vercel domains
echo ""
echo -e "${YELLOW}🌐 Enter your Vercel frontend domains (comma-separated)${NC}"
echo "   Examples: https://stringsight.vercel.app,https://app.stringsight.com"
echo "   Tip: Include both production and preview domains"
read -p "   Enter domains: " VERCEL_DOMAINS_INPUT

# Parse domains into array
IFS=',' read -ra VERCEL_DOMAINS_ARRAY <<< "$VERCEL_DOMAINS_INPUT"

# Build CORS origin header value
if [ ${#VERCEL_DOMAINS_ARRAY[@]} -eq 1 ]; then
    # Single domain - use simple string
    CORS_ORIGIN="${VERCEL_DOMAINS_ARRAY[0]}"
else
    # Multiple domains - Caddy doesn't support multiple origins in one header
    # We'll use the first one as primary
    CORS_ORIGIN="${VERCEL_DOMAINS_ARRAY[0]}"
    echo -e "${YELLOW}⚠️  Note: Using ${CORS_ORIGIN} as primary CORS origin${NC}"
    echo -e "${YELLOW}   For multiple domains, consider using a wildcard or separate matchers${NC}"
fi

# Remove trailing slashes
CORS_ORIGIN="${CORS_ORIGIN%/}"

# Validate it starts with https://
if [[ ! "$CORS_ORIGIN" =~ ^https:// ]]; then
    echo -e "${RED}❌ Domain must start with https://${NC}"
    exit 1
fi

# 5. Write optimized Caddyfile
echo -e "${YELLOW}✍️  Writing optimized Caddyfile...${NC}"
cat > "$CADDY_CONFIG" << CADDY_CONFIG_EOF
# Caddyfile for StringSight API
# Optimized for Vercel → Digital Ocean connections
# Auto-generated on $(date)

api.stringsight.com {
    # Reverse proxy to your uvicorn backend
    reverse_proxy localhost:8000 {
        # Connection pooling - keeps connections alive
        transport http {
            keepalive 90s
            keepalive_idle_conns 10
        }

        # Health check to verify backend is up
        health_uri /health
        health_interval 30s
        health_timeout 5s
    }

    # CORS configuration for Vercel frontend
    @cors_preflight {
        method OPTIONS
    }

    # Handle CORS preflight quickly (this is KEY for performance!)
    handle @cors_preflight {
        header Access-Control-Allow-Origin "$CORS_ORIGIN"
        header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
        header Access-Control-Allow-Headers "DNT, User-Agent, X-Requested-With, If-Modified-Since, Cache-Control, Content-Type, Range, Authorization"
        header Access-Control-Max-Age "86400"  # Cache preflight for 24 hours
        respond 204
    }

    # CORS headers for all responses
    header Access-Control-Allow-Origin "$CORS_ORIGIN"
    header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS"
    header Access-Control-Allow-Headers "DNT, User-Agent, X-Requested-With, If-Modified-Since, Cache-Control, Content-Type, Range, Authorization"
    header Access-Control-Expose-Headers "Content-Length, Content-Range"
    header Access-Control-Allow-Credentials "true"

    # Increase request size limit for file uploads
    request_body {
        max_size 100MB
    }

    # Enable compression for API responses
    encode gzip

    # Logging
    log {
        output file /var/log/caddy/stringsight.log
        format json
        level INFO
    }
}
CADDY_CONFIG_EOF

# 6. Validate Caddyfile
echo -e "${YELLOW}🧪 Validating Caddyfile...${NC}"
if caddy validate --config "$CADDY_CONFIG"; then
    echo -e "${GREEN}✅ Caddyfile is valid${NC}"
else
    echo -e "${RED}❌ Caddyfile validation failed!${NC}"
    echo -e "${YELLOW}💾 Restoring backup...${NC}"
    if [ -f "$BACKUP_FILE" ]; then
        cp "$BACKUP_FILE" "$CADDY_CONFIG"
        echo -e "${GREEN}✅ Backup restored${NC}"
    fi
    exit 1
fi

# 7. Reload Caddy
echo -e "${YELLOW}🔄 Reloading Caddy...${NC}"
if systemctl is-active --quiet caddy; then
    systemctl reload caddy
    echo -e "${GREEN}✅ Caddy reloaded${NC}"
else
    echo -e "${YELLOW}⚠️  Caddy service not running. Starting...${NC}"
    systemctl start caddy
    systemctl enable caddy
    echo -e "${GREEN}✅ Caddy started and enabled${NC}"
fi

# 8. Check Caddy status
sleep 2
if systemctl is-active --quiet caddy; then
    echo -e "${GREEN}✅ Caddy is running${NC}"
else
    echo -e "${RED}❌ Caddy failed to start!${NC}"
    echo -e "${YELLOW}Check logs with: journalctl -u caddy -n 50${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Caddy configuration deployed successfully!${NC}"
echo ""
echo -e "${YELLOW}📊 Next steps:${NC}"
echo "   1. Test health check from your local machine:"
echo "      curl -I https://api.stringsight.com/health"
echo ""
echo "   2. Test CORS preflight:"
echo "      curl -I -X OPTIONS https://api.stringsight.com/health \\"
echo "        -H \"Origin: $CORS_ORIGIN\" \\"
echo "        -H \"Access-Control-Request-Method: GET\""
echo ""
echo "   3. Check Caddy logs:"
echo "      tail -f /var/log/caddy/stringsight.log"
echo "      journalctl -u caddy -f"
echo ""
echo "   4. Deploy your frontend changes to Vercel"
echo ""
echo -e "${GREEN}🎉 Your backend should now connect faster from Vercel!${NC}"
echo ""
echo -e "${YELLOW}💡 Tip: Keep this Caddyfile in your git repo for version control${NC}"
echo "   Git repo Caddyfile: $(pwd)/Caddyfile"
echo "   Active Caddyfile:   $CADDY_CONFIG"
