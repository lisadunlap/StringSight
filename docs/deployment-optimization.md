# Digital Ocean Deployment Optimization Guide

## Problem: Cold Start Issues

When deploying the backend on Digital Ocean with frontend on Vercel, users experience 20-30 second delays before the backend responds, showing "we are having trouble connecting" errors.

## Root Causes

1. **Memory constraints**: Basic droplets ($5-12/mo = 1GB RAM) may kill containers due to OOM
2. **Database connection pooling**: First request after inactivity takes time to establish connections
3. **Container restart policies**: Containers may restart without proper health checks

## Solutions Implemented

### 1. Frontend: Health Check Retry Logic ✅

**File**: `~/stringsight-frontend/src/App.tsx`

The frontend now retries health checks for ~60 seconds with progressive delays:
- Attempt 1: Immediate (0s)
- Attempt 2: After 2s delay
- Attempt 3: After 5s delay
- Attempt 4: After 10s delay
- Attempt 5: After 15s delay
- Attempt 6: After 20s delay

Total retry window: ~52 seconds, which covers typical 20-30s cold starts.

### 2. Backend: PostgreSQL Connection Pooling ✅

**File**: `stringsight/database.py`

Added optimized connection pool settings for PostgreSQL:
```python
pool_size=5           # Keep 5 connections alive
max_overflow=10       # Allow up to 10 additional connections during spikes
pool_pre_ping=True    # Verify connections before use (prevents stale connections)
pool_recycle=3600     # Recycle connections after 1 hour
```

## Recommended Deployment Steps

### Digital Ocean Setup

1. **Upgrade Droplet Size** (Recommended)
   - Minimum: $24/mo Droplet (2GB RAM, 2 vCPUs)
   - Optimal: $48/mo Droplet (4GB RAM, 2 vCPUs)

   This prevents OOM kills and provides headroom for LLM API calls.

2. **Or Add Swap Space** (Budget option)
   ```bash
   # SSH into droplet
   sudo fallocate -l 2G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
   ```

3. **Docker Configuration**

   Create `docker-compose.prod.yml` on your droplet:
   ```yaml
   version: '3.8'

   services:
     db:
       image: postgres:15-alpine
       restart: unless-stopped
       volumes:
         - postgres_data:/var/lib/postgresql/data
       environment:
         - POSTGRES_USER=stringsight
         - POSTGRES_PASSWORD=${DB_PASSWORD}  # Use secure password
         - POSTGRES_DB=stringsight
       healthcheck:
         test: ["CMD-SHELL", "pg_isready -U stringsight"]
         interval: 10s
         timeout: 5s
         retries: 3

     redis:
       image: redis:7-alpine
       restart: unless-stopped
       healthcheck:
         test: ["CMD", "redis-cli", "ping"]
         interval: 10s
         timeout: 5s
         retries: 3

     api:
       image: your-registry/stringsight:latest
       restart: unless-stopped
       ports:
         - "8000:8000"
       environment:
         - DATABASE_URL=postgresql://stringsight:${DB_PASSWORD}@db:5432/stringsight
         - REDIS_URL=redis://redis:6379/0
         - OPENAI_API_KEY=${OPENAI_API_KEY}
       depends_on:
         db:
           condition: service_healthy
         redis:
           condition: service_healthy
       command: uvicorn stringsight.api:app --host 0.0.0.0 --port 8000 --workers 2
       healthcheck:
         test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
         interval: 30s
         timeout: 10s
         retries: 3
         start_period: 40s

     worker:
       image: your-registry/stringsight:latest
       restart: unless-stopped
       environment:
         - DATABASE_URL=postgresql://stringsight:${DB_PASSWORD}@db:5432/stringsight
         - REDIS_URL=redis://redis:6379/0
         - OPENAI_API_KEY=${OPENAI_API_KEY}
       depends_on:
         db:
           condition: service_healthy
         redis:
           condition: service_healthy
       command: celery -A stringsight.celery_app worker --loglevel=info --max-tasks-per-child=10

   volumes:
     postgres_data:
   ```

   Key changes from dev config:
   - `restart: unless-stopped` on all services
   - Removed `--reload` flag (use `--workers 2` instead)
   - Added `start_period: 40s` to health check (allows warm-up time)
   - Secure environment variables from `.env` file

4. **Environment Variables**

   Create `.env` on your droplet:
   ```bash
   DB_PASSWORD=<secure-random-password>
   OPENAI_API_KEY=<your-key>
   ANTHROPIC_API_KEY=<your-key>  # optional
   LOG_LEVEL=INFO
   JSON_LOGS=true
   ```

5. **Deploy**
   ```bash
   # On your droplet
   docker-compose -f docker-compose.prod.yml up -d

   # Check logs
   docker-compose -f docker-compose.prod.yml logs -f api

   # Check container health
   docker ps
   ```

### Vercel Frontend Configuration

Update your Vercel environment variables:

```bash
# In Vercel dashboard → Settings → Environment Variables
VITE_BACKEND=https://your-droplet-ip:8000
```

Or use a domain:
```bash
VITE_BACKEND=https://api.stringsight.com
```

### Nginx Reverse Proxy (Optional but Recommended)

Add nginx for SSL termination and better request handling:

```nginx
# /etc/nginx/sites-available/stringsight
upstream stringsight_api {
    server localhost:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.stringsight.com;

    # Redirect to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.stringsight.com;

    ssl_certificate /etc/letsencrypt/live/api.stringsight.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.stringsight.com/privkey.pem;

    # Increase timeouts for long-running clustering jobs
    proxy_connect_timeout 600s;
    proxy_send_timeout 600s;
    proxy_read_timeout 600s;

    location / {
        proxy_pass http://stringsight_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Enable and restart:
```bash
sudo ln -s /etc/nginx/sites-available/stringsight /etc/nginx/sites-enabled/
sudo certbot --nginx -d api.stringsight.com  # Get SSL cert
sudo nginx -t
sudo systemctl restart nginx
```

## Monitoring & Troubleshooting

### Check for OOM Kills
```bash
# On droplet
dmesg | grep -i "out of memory"
docker stats --no-stream
```

### Monitor Container Health
```bash
docker ps  # Check STATUS column for "healthy"
docker-compose logs -f api --tail 100
```

### Database Connection Issues
```bash
# Check PostgreSQL connections
docker exec -it <db-container> psql -U stringsight -c "SELECT count(*) FROM pg_stat_activity;"

# Should show ~5 idle connections (from pool_size=5)
```

### Cold Start Timing
```bash
# Test backend health from external network
time curl https://api.stringsight.com/health

# Should respond in <1s after warm-up
```

## Expected Performance

After implementing these changes:

- **First request after deploy**: 5-10 seconds (container start + DB connections)
- **First request after 30min idle**: 1-2 seconds (connection pool warm)
- **Subsequent requests**: <500ms

If you still see 20-30 second delays:
1. Check `docker stats` for memory issues
2. Check `docker logs` for startup errors
3. Verify database credentials are correct
4. Ensure Redis is accessible from the API container

## Deploying Updates

### Option 1: Manual Deploy
```bash
# On droplet
git pull origin main
docker-compose -f docker-compose.prod.yml build api worker
docker-compose -f docker-compose.prod.yml up -d --no-deps api worker
```

### Option 2: CI/CD with GitHub Actions
Create `.github/workflows/deploy.yml`:
```yaml
name: Deploy to Digital Ocean

on:
  push:
    branches: [main, prod]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build and push Docker image
        run: |
          docker build -t registry.digitalocean.com/stringsight/api:latest .
          docker push registry.digitalocean.com/stringsight/api:latest

      - name: Deploy to droplet
        uses: appleboy/ssh-action@master
        with:
          host: ${{ secrets.DROPLET_IP }}
          username: root
          key: ${{ secrets.DROPLET_SSH_KEY }}
          script: |
            cd /app/stringsight
            docker-compose -f docker-compose.prod.yml pull
            docker-compose -f docker-compose.prod.yml up -d --no-deps api worker
```

## Cost Optimization

Current setup costs:
- **Basic Droplet ($12/mo)**: May have OOM issues ❌
- **2GB Droplet ($24/mo)**: Recommended minimum ✅
- **4GB Droplet ($48/mo)**: Optimal for production ✅
- **Vercel (Hobby)**: Free tier works fine ✅

Total recommended: $24-48/mo
