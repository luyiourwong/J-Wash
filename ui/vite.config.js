import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev proxy target: the backend port (run.py --port), overridable so a
// non-default instance can be hot-reload developed too.
const backend = `127.0.0.1:${process.env.JWASH_PORT || 8381}`

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': `http://${backend}`,
      '/ws': { target: `ws://${backend}`, ws: true },
    },
  },
})
