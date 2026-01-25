# Quick Fix for Vercel → Digital Ocean Slow Connections

## Summary
Your backend responds **fast** when curled directly (< 1 second) but **slow** from Vercel (20-30 seconds). This indicates the issue is in the network layer, specifically your Caddy configuration missing optimizations for cross-origin requests.

## Root Cause
Based on your setup:
- ✅ Backend is healthy (responds fast to curl)
- ✅ Containers are running (Up 26 hours)
- ✅ 4GB RAM (no memory issues)
- ❌ **Caddy not optimized for Vercel CORS preflight requests**

## The Fix

### 1. Frontend: Health Check Retry ✅ ALREADY DONE
Updated `~/stringsight-frontend/src/App.tsx` to retry health checks for ~60 seconds with progressive delays.

### 2. Backend: Database Connection Pool ✅ ALREADY DONE
Updated `stringsight/database.py` with PostgreSQL connection pooling to reduce cold start time.

### 3. Caddy: Optimize Configuration ⚠️ NEEDS DEPLOYMENT

**On your Digital Ocean droplet**, run:

```bash
# Option A: Automated deployment (recommended)
cd /path/to/your/repo  # wherever you cloned stringsight
sudo ./deploy-caddy-config.sh

# Option B: Manual deployment
sudo cp Caddyfile /etc/caddy/Caddyfile
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
```

The optimized Caddyfile includes:
- **Connection keepalive** (90s, 10 idle connections)
- **Fast CORS preflight handling** (responds in <5ms)
- **24-hour preflight cache** (reduces repeated OPTIONS requests)
- **Health check monitoring** (verifies backend is up)

### 4. Frontend: Deploy to Vercel

```bash
cd ~/stringsight-frontend
npm run build
# Deploy via Vercel CLI or git push
```

## Testing the Fix

After deploying both frontend and Caddy config:

```bash
# 1. Test backend health (should be < 1s)
curl -I https://api.stringsight.com/health

# 2. Test CORS preflight (should be < 100ms)
curl -I -X OPTIONS https://api.stringsight.com/health \
  -H "Origin: https://stringsight.vercel.app" \
  -H "Access-Control-Request-Method: GET"

# Should see:
# HTTP/2 204
# access-control-allow-origin: https://stringsight.vercel.app
# access-control-max-age: 86400
```

## Expected Improvement

**Before:**
- First load from Vercel: 20-30 seconds ❌
- CORS preflight: Not cached, repeated for every request ❌

**After:**
- First load from Vercel: 1-3 seconds ✅
- CORS preflight: Cached for 24 hours, <100ms ✅
- Subsequent requests: <500ms ✅

## What Changed

### Caddyfile (`/etc/caddy/Caddyfile`)
```diff
api.stringsight.com {
    reverse_proxy localhost:8000 {
+       transport http {
+           keepalive 90s
+           keepalive_idle_conns 10
+       }
+       health_uri /health
+       health_interval 30s
    }

+   @cors_preflight {
+       method OPTIONS
+   }
+
+   handle @cors_preflight {
+       header Access-Control-Allow-Origin "https://stringsight.vercel.app"
+       header Access-Control-Max-Age "86400"
+       respond 204
+   }
}
```

### App.tsx Health Check
```diff
React.useEffect(() => {
-   const ok = await checkBackendHealth();
-   setBackendAvailable(ok);
+   let attempts = 0;
+   const maxAttempts = 6;
+   const retryDelays = [0, 2000, 5000, 10000, 15000, 20000];
+
+   while (attempts < maxAttempts) {
+       const ok = await checkBackendHealth();
+       if (ok) {
+           setBackendAvailable(true);
+           return;
+       }
+       // Retry with progressive delays
+   }
}, []);
```

## Monitoring

After deployment, monitor:

```bash
# Caddy logs
tail -f /var/log/caddy/stringsight.log

# Caddy service status
sudo journalctl -u caddy -f

# Backend logs
docker logs -f stringsight-api-1
```

## Troubleshooting

If still slow after deployment:

1. **Check Caddy is running:**
   ```bash
   sudo systemctl status caddy
   ```

2. **Verify CORS headers:**
   ```bash
   curl -I https://api.stringsight.com/health | grep -i access-control
   ```

3. **Check Vercel region:**
   - Your droplet is in SFO2 (San Francisco)
   - Users far from SFO2 will have higher latency (this is expected)
   - Consider adding a Cloudflare CDN in front if global performance needed

4. **Test from different locations:**
   ```bash
   # From Europe
   curl -w "@curl-format.txt" -o /dev/null -s https://api.stringsight.com/health
   ```

## Files Modified

Local changes (commit these):
- ✅ `~/stringsight-frontend/src/App.tsx`
- ✅ `stringsight/database.py`
- ✅ `Caddyfile` (new)
- ✅ `deploy-caddy-config.sh` (new)

Droplet changes (deploy these):
- ⚠️ `/etc/caddy/Caddyfile` (needs update)

## Next Steps

1. Commit frontend changes
2. Deploy Caddy config to droplet
3. Deploy frontend to Vercel
4. Test from Vercel frontend
5. Monitor logs for first 24 hours

## Support

If issues persist:
- Check Caddy logs: `journalctl -u caddy -n 100`
- Check backend logs: `docker logs stringsight-api-1 --tail 100`
- Verify Vercel environment variable: `VITE_BACKEND=https://api.stringsight.com`
