import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const rootDir = dirname(fileURLToPath(import.meta.url))
  const env = loadEnv(mode, rootDir, '')

  return {
    plugins: [react()],
    server: {
      host: true,
      port: 5180,
      proxy: {
        // Proxy API requests to the backend to avoid CORS and ad-blockers
        '/api': {
          target: env.VITE_BACKEND || 'http://localhost:8000',
          changeOrigin: true,
          rewrite: (path: string) => path.replace(/^\/api/, ''),
        },
      },
    },
  }
})
