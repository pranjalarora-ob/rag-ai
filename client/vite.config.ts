import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const rootEnv = loadEnv(mode, '..', '')
  const clientEnv = loadEnv(mode, '.', '')
  const proxyTarget =
    clientEnv.API_PROXY_TARGET ||
    clientEnv.VITE_API_PROXY_TARGET ||
    rootEnv.API_PROXY_TARGET ||
    rootEnv.VITE_API_PROXY_TARGET ||
    `http://localhost:${rootEnv.PORT || 3000}`

  return {
    plugins: [react()],
    server: {
      port: 5173,
      proxy: {
        '/rag': {
          target: proxyTarget,
          changeOrigin: true,
        },
      },
    },
  }
})
