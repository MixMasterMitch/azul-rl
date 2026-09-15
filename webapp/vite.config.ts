import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig(({ mode }) => ({
  plugins: [react()],
  server: {
    proxy: {
      '/api': loadEnv(mode, '.', 'AZUL_').AZUL_API_PROXY || 'http://localhost:5000',
    },
  },
}))
