import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/reconstruct': 'http://localhost:8000',
      '/search':      'http://localhost:8000',
      '/interpolate': 'http://localhost:8000',
      '/generate':    'http://localhost:8000',
      '/tsne':        'http://localhost:8000',
    },
  },
})
